
#include "cublas_utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define STB_IMAGE_IMPLEMENTATION
#include "dirent.h"
#include "stb_image.h"
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

cublasHandle_t cublasH = NULL;

// input和output均为(batch_size x n)矩阵（列优先） bias为长度为n的向量
__global__ void addBiasAndReLU(float* input, float* output, float* bias, int m, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n)
    {
        int row = idx % m;
        int col = idx / m;
        output[col * m + row] = max(0.0f, input[col * m + row] + bias[col]);
    }
}

__global__ void addBias(float* input, float* output, float* bias, int m, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n)
    {
        int row = idx % m;
        int col = idx / m;
        output[col * m + row] = input[col * m + row] + bias[col];
    }
}

__global__ void calcDeltaLoss(float* input, float* output, float* labels, int batchSize, int k)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample < batchSize)
    {
        float sum = 0;
        for (int i = 0; i < k; i++)
        {
            sum += exp(input[i * batchSize + sample]);
        }
        for (int i = 0; i < k; i++)
        {
            float pi = exp(input[i * batchSize + sample]) / sum;
            // labels转one-hot编码
            output[i * batchSize + sample] = (pi - (labels[sample] == i ? 1 : 0)) / batchSize;
        }
    }
}

__global__ void odotIndex(float* input, float* output, float* cacheZ, int m, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n)
    {
        int batch = idx % m;
        int cell = idx / m;
        output[cell * m + batch] = input[cell * m + batch] * (cacheZ[cell * m + batch] > 0 ? 1 : 0);
    }
}

__global__ void updateModel(float* weights, float* bias, int inputSize, int width, int batch_size)
{

}

// input是一个矩阵(batchSize x width) 行优先
// 超长参数 感觉应该抽象一下的
static void forwardPass(float* input, float* output, float* cacheA, float* cacheZ, float* weights, float* bias,
                        int inputSize, int width, int batchSize, bool relu, bool transA)
{
    // 先线性变换
    float alpha = 1.0f;
    float beta = 0.0f;
    // printf("%d %d\n", batchSize, width);
    //  因为cublas是列优先的，weights这里隐含一个转置操作，这刚好有利于计算
    if (transA)
    {
        cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, batchSize, width, inputSize, &alpha, input, inputSize, weights,
                    inputSize, &beta, output, batchSize);
    }
    else
    {
        cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, width, inputSize, &alpha, input, batchSize, weights,
                    inputSize, &beta, output, batchSize);
    }
    // cacheZ: batchSize x vec(z)
    cudaMemcpy(cacheZ, output, batchSize * width * sizeof(float), cudaMemcpyDeviceToDevice);

    if (relu)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (batchSize * width + threadsPerBlock - 1) / threadsPerBlock;
        addBiasAndReLU<<<blocksPerGrid, threadsPerBlock>>>(output, output, bias, batchSize, width);
    }
    else
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (batchSize * width + threadsPerBlock - 1) / threadsPerBlock;
        addBias<<<blocksPerGrid, threadsPerBlock>>>(output, output, bias, batchSize, width);
    }
    // cacheA: batchSize x vec(a)
    cudaMemcpy(cacheA, output, batchSize * width * sizeof(float), cudaMemcpyDeviceToDevice);
}

// 计算所有L对Z的微分
// 这里的gradients和lastGradient都是L对z的微分的批处理矩阵
static void backwardPass(float* gradients, float* lastGradient, float* cacheZ, float* weights, int inputSize,
                         int outputSize, int lastOutputSize, int batchSize)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    // output是L对A的微分
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, outputSize, lastOutputSize, &alpha, lastGradient,
                batchSize, weights, outputSize, &beta, gradients, batchSize);
    // output是L对Z的微分 batchSize x L'(vec(z))
    int threadsPerBlock = 256;
    int blocksPerGrid = (batchSize * outputSize + threadsPerBlock - 1) / threadsPerBlock;
    odotIndex<<<blocksPerGrid, threadsPerBlock>>>(gradients, gradients, cacheZ, batchSize, outputSize);
}

enum ErrorCode
{
    SUCCESS = 0,
    FILE_NOT_FOUND,
    INVALID_FORMAT,
    INTERNAL_ERROR
};

static inline int checkCUDA(cudaError_t err)
{
    if (err != 0)
    {
        printf("CUDA ERROR:%d\n", err);
        return 1;
    }
    return 0;
}

/*
在MNIST任务中
x是特征矩阵，每一行是一个样本
y是标签向量，每一行是一个样本对应的标签
size是样本数目
stride是每一行的长度
*/
typedef struct
{
    float* x;
    float* y;
    int size = -1;
    int stride = -1;
} DataSet;

typedef struct
{
    // host
    int inputSize;
    int outputSize;
    int offsetWeights;
    int offsetBiases;
    int width;
} Layer;

/*
在MNIST任务中 一个batch的
features是一个矩阵 labels是一个向量
size是样本数目
*/
typedef struct
{
    float* features;
    float* labels;
    int size;
} Batch;

/*
隐藏层均使用线性变换和ReLU激活函数
用于MNIST的MLP
每个隐藏层的输入和输出都是向量
在批处理中拓展成矩阵
*/
class MLP
{
  private:
    // device
    float* weights;
    float* gradients;
    float* biases;
    float* cacheA;
    float* cacheZ;
    float* inputTemp;
    float* outputTemp;
    // host
    Layer* layers;
    Layer outLayer;
    const int depth;
    const int inputSize;
    const int outputSize;
    const int batchSize;

    int layerIndex = 0;
    int offsetWeights = 0;
    int offsetBiases = 0;
    int maxWidth = 0;

  public:
    MLP(int depth, int inputSize, int outputSize, int batchSize)
        : depth(depth), inputSize(inputSize), outputSize(outputSize), batchSize(batchSize)
    {
        layers = new Layer[depth];
        if (inputSize > maxWidth)
        {
            maxWidth = inputSize;
        }
        if (outputSize > maxWidth)
        {
            maxWidth = outputSize;
        }
    }

    /*
    添加一个有width个神经元的隐藏层
    */
    ErrorCode addLayer(int width)
    {
        if (layerIndex >= depth)
        {
            return INTERNAL_ERROR;
        }
        if (layerIndex == 0)
        {
            layers[layerIndex].inputSize = inputSize;
        }
        else
        {
            layers[layerIndex].inputSize = layers[layerIndex - 1].outputSize;
        }
        layers[layerIndex].outputSize = width;
        layers[layerIndex].offsetWeights = offsetWeights;
        layers[layerIndex].offsetBiases = offsetBiases;
        offsetWeights += width * layers[layerIndex].inputSize;
        offsetBiases += width;
        if (width > maxWidth)
        {
            maxWidth = width;
        }
        layerIndex++;
        return SUCCESS;
    }

    ErrorCode build()
    {
        outLayer.inputSize = layers[depth - 1].outputSize;
        outLayer.outputSize = outputSize;
        outLayer.offsetWeights = offsetWeights;
        outLayer.offsetBiases = offsetBiases;
        offsetWeights += outputSize * outLayer.inputSize;
        offsetBiases += outputSize;

        int totalWeights = offsetWeights;
        int totalBiases = offsetBiases;
        int flag = 0;
        flag |= checkCUDA(cudaMalloc(&weights, totalWeights * sizeof(float)));
        flag |= checkCUDA(cudaMalloc(&biases, totalBiases * sizeof(float)));
        // 用于存储所有层的L对z的微分的批处理矩阵 注意不是L对w和L对b
        flag |= checkCUDA(cudaMalloc(&gradients, batchSize * totalBiases * sizeof(float)));
        flag |= checkCUDA(cudaMalloc(&cacheA, batchSize * totalBiases * sizeof(float)));
        flag |= checkCUDA(cudaMalloc(&cacheZ, batchSize * totalBiases * sizeof(float)));
        flag |= checkCUDA(cudaMalloc(&inputTemp, batchSize * maxWidth * sizeof(float)));
        flag |= checkCUDA(cudaMalloc(&outputTemp, batchSize * maxWidth * sizeof(float)));
        if (flag)
        {
            return INTERNAL_ERROR;
        }
        return SUCCESS;
    }

    // 一个iterate包含前向传播、反向传播和参数更新
    // 链式微分还不太熟悉 会有点混乱 查阅资料辅助：
    // 从输出层开始，反向逐层计算：
    //     1.先求损失函数对当前层输出的梯度
    //     2.再用这个梯度求当前层参数的梯度
    //     3.然后继续传播到上一层
    // 这里是简单的串行了，查阅资料发现这种串行可以优化，例如：
    // # 实际执行时的重叠计算（以时间线表示）
    //     时间 | GPU活动
    //     ---------------------------- -
    //     t = 0 | Batch1: 前向传播
    //     t = 1 | Batch1 : 反向传播 | Batch2 : 数据加载到GPU
    //     t = 2 | Batch1 : 参数更新 | Batch2 : 前向传播
    //     t = 3 | Batch2 : 反向传播 | Batch3 : 数据加载
    //     t = 4 | Batch2 : 参数更新 | Batch3 : 前向传播
    ErrorCode iterate(Batch batch)
    {
        int cacheOffset = 0;
        float* temp;
        // L对logits的平均微分 1/N*(P-Y)
        forwardPass(batch.features, outputTemp, cacheA + cacheOffset, cacheZ + cacheOffset,
                    weights + layers[0].offsetWeights, biases + layers[0].offsetBiases, layers[0].inputSize,
                    layers[0].outputSize, batch.size, true, true);
        temp = outputTemp;
        outputTemp = inputTemp;
        inputTemp = temp;
        cacheOffset += batch.size * layers[0].outputSize;
        for (int i = 1; i < depth; i++)
        {
            forwardPass(inputTemp, outputTemp, cacheA + cacheOffset, cacheZ + cacheOffset,
                        weights + layers[i].offsetWeights, biases + layers[i].offsetBiases, layers[i].inputSize,
                        layers[i].outputSize, batch.size, true, false);
            temp = outputTemp;
            outputTemp = inputTemp;
            inputTemp = temp;
            cacheOffset += batch.size * layers[i].outputSize;
        }
        forwardPass(inputTemp, outputTemp, cacheA + cacheOffset, cacheZ + cacheOffset, weights + outLayer.offsetWeights,
                    biases + outLayer.offsetBiases, outLayer.inputSize, outLayer.outputSize, batch.size, false, false);
        temp = outputTemp;
        outputTemp = inputTemp;
        inputTemp = temp;
        int threadsPerBlock = 256;
        int blocksPerGrid = (batch.size + threadsPerBlock - 1) / threadsPerBlock;
        calcDeltaLoss<<<blocksPerGrid, threadsPerBlock>>>(inputTemp, outputTemp, batch.labels, batch.size,
                                                          outLayer.outputSize);
        cudaMemcpy(gradients + cacheOffset, outputTemp, batch.size * outLayer.outputSize * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cacheOffset -= batch.size * layers[depth - 1].outputSize;
        temp = outputTemp;
        outputTemp = inputTemp;
        inputTemp = temp;
        for (int i = depth - 1; i >= 0; i--)
        {
            backwardPass(outputTemp, inputTemp, cacheZ + cacheOffset, weights + layers[i].offsetWeights,
                         layers[i].inputSize, layers[i].outputSize,
                         (i == depth - 1) ? outLayer.outputSize : layers[i + 1].outputSize, batchSize);
            cudaMemcpy(gradients + cacheOffset, outputTemp, batch.size * layers[i].outputSize * sizeof(float),
                       cudaMemcpyDeviceToDevice);
            temp = outputTemp;
            outputTemp = inputTemp;
            inputTemp = temp;
            cacheOffset -= batch.size * layers[i].outputSize;
        }



        return SUCCESS;
    }

    ErrorCode train(DataSet* trainset)
    {
        int numBatches = (trainset->size + batchSize - 1) / batchSize;
        // for (int b = 0; b < 1; b++)
        for (int b = 0; b < numBatches; b++)
        {
            int currentBatchSize =
                ((b + 1) * batchSize <= trainset->size) ? batchSize : (trainset->size - b * batchSize);
            Batch batch;
            batch.size = currentBatchSize;
            batch.features = trainset->x + b * batchSize * trainset->stride;
            batch.labels = trainset->y + b * batchSize;
            iterate(batch);
        }
        return SUCCESS;
    }

    ErrorCode test(DataSet* testset)
    {
        return SUCCESS;
    }

    void free()
    {
        cudaFree(weights);
        cudaFree(gradients);
        cudaFree(biases);
        cudaFree(cacheA);
        cudaFree(cacheZ);
        cudaFree(inputTemp);
        cudaFree(outputTemp);
        delete[] layers;
    }
};

// 考虑一下训练结果和训练集label无序性的关系
// todo 实现一：不打乱 实现二：打乱
ErrorCode readDataSet(DataSet* dest, const char* dir)
{
    std::vector<float> data;
    std::vector<float> labels;
    printf("loading dataset from %s\n", dir);
    unsigned long long cap = 0;
    // prehandle
    for (int type = 0; type <= 9; type++)
    {
        int count = 0;
        std::string path = std::string(dir) + std::to_string(type);
        DIR* dir = opendir(path.c_str());
        if (dir == nullptr)
            return FILE_NOT_FOUND;

        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr)
        {
            if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, ".."))
            {
                count++;
            }
        }
        closedir(dir);
        cap += count;
        printf("type %d : %d samples loaded\n", type, count);
    }
    // trick 根据MNIST任务 硬编码优化
    dest->stride = 28 * 28;
    data.reserve(cap * dest->stride);
    labels.reserve(cap);
    // load
    int counter = 0;
    for (int type = 0; type <= 9; type++)
    {
        float inv = 1.0f / 255.0f;
        std::string path = std::string(dir) + std::to_string(type);
        DIR* dir = opendir(path.c_str());
        if (dir == nullptr)
            return FILE_NOT_FOUND;

        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr)
        {
            if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, ".."))
            {
                std::string fileName = path + "/" + entry->d_name;
                int width, height, channel;
                unsigned char* img = stbi_load(fileName.c_str(), &width, &height, &channel, 1);
                int index = 0;
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        data.push_back(static_cast<float>(img[index]) * inv);
                        index++;
                    }
                }
                labels.push_back(static_cast<float>(type));
                stbi_image_free(img);
                if (index == 0)
                {
                    return INTERNAL_ERROR;
                }
                counter++;
            }
        }
        printf("\r%d / %llu", counter, cap);
        closedir(dir);
    }
    printf("\n");
    int cudaState = 0;
    cudaState |= checkCUDA(cudaMalloc(&dest->x, data.size() * sizeof(float)));
    cudaState |= checkCUDA(cudaMalloc(&dest->y, labels.size() * sizeof(float)));
    cudaState |= checkCUDA(cudaMemcpy(dest->x, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));
    cudaState |= checkCUDA(cudaMemcpy(dest->y, labels.data(), labels.size() * sizeof(float), cudaMemcpyHostToDevice));
    if (cudaState)
    {
        return INTERNAL_ERROR;
    }
    dest->size = static_cast<int>(labels.size());
    printf("read result: %d samples loaded.\n", dest->size);
    return SUCCESS;
}

constexpr int MODEL_DEPTH = 3;
constexpr int INPUT_SIZE = 28 * 28;
constexpr int OUTPUT_SIZE = 10;
constexpr int BATCH_SIZE = 64;

DataSet trainSet;
DataSet testSet;
MLP model(MODEL_DEPTH, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE);
int main()
{
    CUBLAS_CHECK(cublasCreate(&cublasH));
    // READ
    ErrorCode readState1 = readDataSet(&trainSet, "E:/study/mnist/python/output/train/");
    printf("check readState:%d\n", readState1);
    ErrorCode readState2 = readDataSet(&testSet, "E:/study/mnist/python/output/test/");
    printf("check readState:%d\n", readState2);
    // TRAIN
    int modelBuildState = SUCCESS;
    modelBuildState |= model.addLayer(256);
    modelBuildState |= model.addLayer(128);
    modelBuildState |= model.addLayer(64);
    model.build();
    printf("check modelBuildState:%d\n", modelBuildState);
    ErrorCode modelTrainState = model.train(&trainSet);
    printf("check modelTrainState:%d\n", modelTrainState);
    // TEST
    CUBLAS_CHECK(cublasDestroy(cublasH));
    cudaDeviceReset();
    model.free();
    return 0;
}
