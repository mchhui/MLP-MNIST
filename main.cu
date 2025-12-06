
#include "cublas_utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define STB_IMAGE_IMPLEMENTATION
#include "dirent.h"
#include "stb_image.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

cublasHandle_t cublasH = NULL;

// src是行优先排列的 dest是列优先排列的 m,n:src和dest的行数和列数
__global__ void transpose(float* dest, float* src, int m, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n)
    {
        int row = idx / n;
        int col = idx % n;
        dest[col * m + row] = src[row * n + col];
    }
}

/*
input:列优先 batchSize x width
output:列优先 batchSize x width
bias:长度为width的向量
*/
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

__global__ void applyReLU(float* input, float* output, int m, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n)
    {
        int row = idx % m;
        int col = idx / m;
        output[col * m + row] = max(0.0f, input[col * m + row]);
    }
}

__global__ void softmax(float* input, float* output, int batchSize, int k)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample < batchSize)
    {
        // 找到最大值以实现数值稳定
        float max_val = input[0 * batchSize + sample];
        for (int i = 1; i < k; i++)
        {
            max_val = max(max_val, input[i * batchSize + sample]);
        }

        // 计算稳定的指数和（减去max_val以避免数值溢出）
        float sum = 0.0f;
        for (int i = 0; i < k; i++)
        {
            sum += exp(input[i * batchSize + sample] - max_val);
        }

        // 计算softmax概率
        for (int i = 0; i < k; i++)
        {
            output[i * batchSize + sample] = exp(input[i * batchSize + sample] - max_val) / sum;
        }
    }
}
/*
input: 列优先 batchSize x k
output: 列优先 batchSize x k
*/
__global__ void calcDeltaLoss(float* input, float* output, float* labels, int batchSize, int k)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample < batchSize)
    {
        // 找到最大值以实现数值稳定（与softmax保持一致）
        float max_val = input[0 * batchSize + sample];
        for (int i = 1; i < k; i++)
        {
            max_val = max(max_val, input[i * batchSize + sample]);
        }

        // 计算稳定的指数和与
        float sum = 0.0f;
        for (int i = 0; i < k; i++)
        {
            sum += exp(input[i * batchSize + sample] - max_val);
        }

        // 计算交叉熵损失的梯度
        int true_label = static_cast<int>(labels[sample]);
        for (int i = 0; i < k; i++)
        {
            float pi = exp(input[i * batchSize + sample] - max_val) / sum;
            output[i * batchSize + sample] = (pi - (i == true_label ? 1.0f : 0.0f));
        }
    }
}

// 计算交叉熵损失
__global__ void calcCrossEntropyLoss(float* logits, float* labels, float* losses, int batchSize, int k)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample < batchSize)
    {
        // 找到最大值以实现数值稳定
        float max_val = logits[0 * batchSize + sample];
        for (int i = 1; i < k; i++)
        {
            max_val = max(max_val, logits[i * batchSize + sample]);
        }

        // 计算稳定的指数和
        float sum = 0.0f;
        for (int i = 0; i < k; i++)
        {
            sum += exp(logits[i * batchSize + sample] - max_val);
        }

        // 计算交叉熵损失: -log(p_true_label)
        int true_label = static_cast<int>(labels[sample]);

        // 边界检查：确保 true_label 在有效范围内
        if (true_label < 0 || true_label >= k)
        {
            losses[sample] = 1e10f; // 无效标签，设置一个很大的损失值
            return;
        }

        // 添加小的 epsilon 值避免 log(0) 或 sum 为 0 的情况
        const float epsilon = 1e-8f;
        sum = max(sum, epsilon);

        float log_prob = logits[true_label * batchSize + sample] - max_val - logf(sum);
        losses[sample] = -log_prob;
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

__global__ void calcBias(float* bias, float* gradients, int m, int n, float learningRateDivBatchSize)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n)
    {
        float sum = 0;
        for (int i = 0; i < m; i++)
        {
            sum += gradients[col * m + i];
        }
        // 使用梯度下降：bias = bias - learning_rate * gradient
        bias[col] -= sum * learningRateDivBatchSize;
    }
}

/*
input:列优先 batchSize x vec(x)
weights:列优先 width x inputSize
output:列优先 batchSize x width
cacheA:列优先 batchSize x vec(a)
cacheZ:列优先 batchSize x vec(z)
bias:长度为width的向量
*/
static void forwardPass(float* input, float* output, float* cacheA, float* cacheZ, float* weights, float* bias,
                        int inputSize, int width, int batchSize, bool relu)
{
    // 先线性变换
    float alpha = 1.0f;
    float beta = 0.0f;
    // printf("%d %d\n", batchSize, width);
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, batchSize, width, inputSize, &alpha, input, batchSize, weights,
                width, &beta, output, batchSize);

    int threadsPerBlock = 256;
    int blocksPerGrid = (batchSize * width + threadsPerBlock - 1) / threadsPerBlock;

    // 添加偏置，得到z = Wx + b
    addBias<<<blocksPerGrid, threadsPerBlock>>>(output, output, bias, batchSize, width);
    // cacheZ: batchSize x vec(z) = Wx + b (激活前的值)
    cudaMemcpy(cacheZ, output, batchSize * width * sizeof(float), cudaMemcpyDeviceToDevice);

    if (relu)
    {
        // 应用ReLU，得到a = ReLU(z)
        applyReLU<<<blocksPerGrid, threadsPerBlock>>>(output, output, batchSize, width);
    }
    // cacheA: batchSize x vec(a) = 激活后的值
    cudaMemcpy(cacheA, output, batchSize * width * sizeof(float), cudaMemcpyDeviceToDevice);
}

// 计算所有L对Z的微分
// 这里的gradients和lastGradient都是L对z的微分的批处理矩阵
/*
gradients: 列优先 batchSize x outputSize
lastGradient: 列优先 batchSize x lastOutputSize
cacheZ: 列优先 batchSize x outputSize
*/
static void backwardPass(float* gradients, float* lastGradient, float* cacheZ, float* lastWeights, int inputSize,
                         int outputSize, int lastOutputSize, int batchSize)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    // output是L对A的微分
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, outputSize, lastOutputSize, &alpha, lastGradient,
                batchSize, lastWeights, lastOutputSize, &beta, gradients, batchSize);
    // output是L对Z的微分 batchSize x L'(vec(z))
    int threadsPerBlock = 256;
    int blocksPerGrid = (batchSize * outputSize + threadsPerBlock - 1) / threadsPerBlock;
    odotIndex<<<blocksPerGrid, threadsPerBlock>>>(gradients, gradients, cacheZ, batchSize, outputSize);
}

/*
weights: 列优先 width x inputSize
bias: 长度为width的向量
gradients: 列优先 batchSize x width
nextCacheA: 列优先 batchSize x inputSize
*/
static void updateModel(float* weights, float* bias, float* gradients, float* nextCacheA, int inputSize, int width,
                        int batchSize, float alpha)
{
    // 学习率（传入的alpha已经是正数，需要取负以实现梯度下降）
    float negative_alpha = -alpha;
    float beta = 1.0f;
    // A_{i-1}的倒置与L对z_i的微分的乘积
    // vec(w) x width
    cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, width, inputSize, batchSize, &negative_alpha, gradients, batchSize,
                nextCacheA, batchSize, &beta, weights, width);
    int threadsPerBlock = 256;
    int blocksPerGrid = (width + threadsPerBlock - 1) / threadsPerBlock;
    calcBias<<<blocksPerGrid, threadsPerBlock>>>(bias, gradients, batchSize, width, alpha);
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

    float learningRate = 0.01f;
    int epochAmount = 10;

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
        // 发现这里高度对齐
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

        // 使用优化的初始化方案
        std::random_device rd;
        std::mt19937 gen(rd());

        // 初始化所有权重（按层使用合适的初始化方法）
        float* host_weights = new float[totalWeights];
        int weightIdx = 0;

        // 初始化隐藏层权重（使用He初始化，适合ReLU激活函数）
        for (int i = 0; i < depth; i++)
        {
            int fan_in = layers[i].inputSize;
            // He初始化（均匀分布版本）：U(-sqrt(6/fan_in), sqrt(6/fan_in))
            // 这能保持前向和反向传播中的梯度方差
            float limit = sqrtf(6.0f / fan_in);
            std::uniform_real_distribution<float> dis(-limit, limit);

            int layerWeightSize = layers[i].outputSize * layers[i].inputSize;
            for (int j = 0; j < layerWeightSize; j++)
            {
                host_weights[weightIdx++] = dis(gen);
            }
        }

        // 初始化输出层权重（使用Xavier/Glorot初始化，因为输出层没有激活函数）
        int fan_in = outLayer.inputSize;
        int fan_out = outLayer.outputSize;
        // Xavier初始化：U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
        float limit = sqrtf(6.0f / (fan_in + fan_out));
        std::uniform_real_distribution<float> outDis(-limit, limit);

        int outLayerWeightSize = outLayer.outputSize * outLayer.inputSize;
        for (int j = 0; j < outLayerWeightSize; j++)
        {
            host_weights[weightIdx++] = outDis(gen);
        }

        cudaMemcpy(weights, host_weights, totalWeights * sizeof(float), cudaMemcpyHostToDevice);
        delete[] host_weights;

        float* host_biases = new float[totalBiases];
        std::uniform_real_distribution<float> biasDis(-0.01f, 0.01f);
        for (int i = 0; i < totalBiases; i++)
        {
            host_biases[i] = biasDis(gen);
        }
        cudaMemcpy(biases, host_biases, totalBiases * sizeof(float), cudaMemcpyHostToDevice);
        delete[] host_biases;

        printf("权重和偏置初始化完成（He/Xavier初始化）\n");

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
        int flag = 0;
        int cacheOffset = 0;
        float* temp;

        // L对logits的平均微分 1/N*(P-Y)
        {
            int threadsPerBlock = 256;
            int blocksPerGrid = (batch.size * inputSize + threadsPerBlock - 1) / threadsPerBlock;
            transpose<<<blocksPerGrid, threadsPerBlock>>>(inputTemp, batch.features, batch.size, inputSize);
        }
        for (int i = 0; i < depth; i++)
        {
            cacheOffset = batchSize * layers[i].offsetBiases;
            forwardPass(inputTemp, outputTemp, cacheA + cacheOffset, cacheZ + cacheOffset,
                        weights + layers[i].offsetWeights, biases + layers[i].offsetBiases, layers[i].inputSize,
                        layers[i].outputSize, batch.size, true);
            flag |= checkCUDA(cudaGetLastError());

            temp = outputTemp;
            outputTemp = inputTemp;
            inputTemp = temp;
        }

        cacheOffset = batchSize * outLayer.offsetBiases;
        forwardPass(inputTemp, outputTemp, cacheA + cacheOffset, cacheZ + cacheOffset, weights + outLayer.offsetWeights,
                    biases + outLayer.offsetBiases, outLayer.inputSize, outLayer.outputSize, batch.size, false);
        flag |= checkCUDA(cudaGetLastError());

        temp = outputTemp;
        outputTemp = inputTemp;
        inputTemp = temp;

        // 计算梯度
        int threadsPerBlock = 256;
        int blocksPerGrid = (batch.size + threadsPerBlock - 1) / threadsPerBlock;
        calcDeltaLoss<<<blocksPerGrid, threadsPerBlock>>>(inputTemp, outputTemp, batch.labels, batch.size,
                                                          outLayer.outputSize);
        flag |= checkCUDA(cudaGetLastError());

        flag |= checkCUDA(cudaMemcpy(gradients + cacheOffset, outputTemp,
                                     batch.size * outLayer.outputSize * sizeof(float), cudaMemcpyDeviceToDevice));

        temp = outputTemp;
        outputTemp = inputTemp;
        inputTemp = temp;

        for (int i = depth - 1; i >= 0; i--)
        {
            cacheOffset = batchSize * layers[i].offsetBiases;
            float* weightPtr = weights + outLayer.offsetWeights;
            if (i < depth - 1)
            {
                weightPtr = weights + layers[i + 1].offsetWeights;
            }
            backwardPass(outputTemp, inputTemp, cacheZ + cacheOffset, weightPtr, layers[i].inputSize,
                         layers[i].outputSize, (i == depth - 1) ? outLayer.outputSize : layers[i + 1].outputSize,
                         batch.size);
            flag |= checkCUDA(cudaGetLastError());

            flag |= checkCUDA(cudaMemcpy(gradients + cacheOffset, outputTemp,
                                         batch.size * layers[i].outputSize * sizeof(float), cudaMemcpyDeviceToDevice));

            temp = outputTemp;
            outputTemp = inputTemp;
            inputTemp = temp;
        }

        for (int i = 0; i < depth; i++)
        {
            cacheOffset = layers[i].offsetBiases * batchSize;
            float* ptrA = batch.features;
            if (i > 0)
            {
                ptrA = cacheA + layers[i - 1].offsetBiases * batchSize;
            }
            updateModel(weights + layers[i].offsetWeights, biases + layers[i].offsetBiases, gradients + cacheOffset,
                        ptrA, layers[i].inputSize, layers[i].outputSize, batch.size, learningRate / batch.size);
            flag |= checkCUDA(cudaGetLastError());
        }

        cacheOffset = outLayer.offsetBiases * batchSize;
        float* ptrA = cacheA + layers[depth - 1].offsetBiases * batchSize;
        updateModel(weights + outLayer.offsetWeights, biases + outLayer.offsetBiases, gradients + cacheOffset, ptrA,
                    outLayer.inputSize, outLayer.outputSize, batch.size, learningRate / batch.size);
        flag |= checkCUDA(cudaGetLastError());

        return (ErrorCode)flag;
    }

    ErrorCode train(DataSet* trainset)
    {
        int flag = 0;
        for (int i = 0; i < epochAmount; i++)
        {
            int numBatches = (trainset->size + batchSize - 1) / batchSize;
            for (int b = 0; b < numBatches; b++)
            {
                int currentBatchSize = std::min(batchSize, trainset->size - b * batchSize);
                Batch batch;
                batch.size = currentBatchSize;
                batch.features = trainset->x + b * batchSize * trainset->stride;
                batch.labels = trainset->y + b * batchSize;
                flag |= iterate(batch);
            }
        }
        return (ErrorCode)flag;
    }

    ErrorCode test(DataSet* testset)
    {
        int flag = 0;
        int totalCorrect = 0;
        int totalSamples = 0;
        float totalLoss = 0.0f;

        printf("\n开始测试...\n");

        // 分配临时内存用于存储预测结果、损失和标签
        float* hostPredictions = new float[batchSize * outLayer.outputSize];
        float* hostLosses = new float[batchSize];
        float* hostLabels = new float[batchSize];
        float* deviceLosses = nullptr;
        checkCUDA(cudaMalloc(&deviceLosses, batchSize * sizeof(float)));

        int numBatches = (testset->size + batchSize - 1) / batchSize;

        for (int b = 0; b < numBatches; b++)
        {
            int currentBatchSize = std::min(batchSize, testset->size - b * batchSize);
            Batch batch;
            batch.size = currentBatchSize;
            batch.features = testset->x + b * batchSize * testset->stride;
            batch.labels = testset->y + b * batchSize;

            // 前向传播（只做前向，不做反向和更新）
            int cacheOffset = 0;
            float* temp;

            {
                int threadsPerBlock = 256;
                int blocksPerGrid = (batch.size * inputSize + threadsPerBlock - 1) / threadsPerBlock;
                transpose<<<blocksPerGrid, threadsPerBlock>>>(inputTemp, batch.features, batch.size, inputSize);
            }
            // 第一层
            forwardPass(inputTemp, outputTemp, cacheA + cacheOffset, cacheZ + cacheOffset,
                        weights + layers[0].offsetWeights, biases + layers[0].offsetBiases, layers[0].inputSize,
                        layers[0].outputSize, batch.size, true);
            flag |= checkCUDA(cudaGetLastError());

            temp = outputTemp;
            outputTemp = inputTemp;
            inputTemp = temp;

            // 隐藏层
            for (int i = 1; i < depth; i++)
            {
                cacheOffset = batchSize * layers[i].offsetBiases;
                forwardPass(inputTemp, outputTemp, cacheA + cacheOffset, cacheZ + cacheOffset,
                            weights + layers[i].offsetWeights, biases + layers[i].offsetBiases, layers[i].inputSize,
                            layers[i].outputSize, batch.size, true);
                flag |= checkCUDA(cudaGetLastError());

                temp = outputTemp;
                outputTemp = inputTemp;
                inputTemp = temp;
            }

            // 输出层
            cacheOffset = batchSize * outLayer.offsetBiases;
            forwardPass(inputTemp, outputTemp, cacheA + cacheOffset, cacheZ + cacheOffset,
                        weights + outLayer.offsetWeights, biases + outLayer.offsetBiases, outLayer.inputSize,
                        outLayer.outputSize, batch.size, false);
            flag |= checkCUDA(cudaGetLastError());
            temp = outputTemp;
            outputTemp = inputTemp;
            inputTemp = temp;
            // 计算softmax得到预测概率（outputTemp是logits，inputTemp是softmax概率）
            int threadsPerBlock = 256;
            int blocksPerGrid = (batch.size + threadsPerBlock - 1) / threadsPerBlock;
            softmax<<<blocksPerGrid, threadsPerBlock>>>(inputTemp, outputTemp, batch.size, outLayer.outputSize);
            flag |= checkCUDA(cudaGetLastError());
            if (b == 0)
            {
                // 在计算softmax之前，保存logits和labels到文件
                static bool firstWrite = true;
                std::ofstream logitsFile("test_logits_labels.txt", firstWrite ? std::ios::out : std::ios::app);
                if (logitsFile.is_open())
                {
                    // 第一行写表头注释
                    if (firstWrite)
                    {
                        logitsFile << "# Format: label, logit0, logit1, ..., logit9 (列优先格式)\n";
                        firstWrite = false;
                    }

                    // 从GPU复制logits和labels到主机
                    // outputTemp是列优先格式：batchSize x outputSize，但实际只用了batch.size行
                    float* hostLogits = new float[batchSize * outLayer.outputSize];
                    float* hostBatchLabels = new float[batch.size];
                    checkCUDA(cudaMemcpy(hostLogits, outputTemp, batch.size * outLayer.outputSize * sizeof(float),
                                         cudaMemcpyDeviceToHost));
                    checkCUDA(
                        cudaMemcpy(hostBatchLabels, batch.labels, batch.size * sizeof(float), cudaMemcpyDeviceToHost));

                    // 写入文件（列优先格式：第i个样本的第j个logit在 hostLogits[j * batchSize + i]）
                    logitsFile << std::fixed << std::setprecision(6);
                    for (int i = 0; i < batch.size; i++)
                    {
                        logitsFile << static_cast<int>(hostBatchLabels[i]);
                        for (int j = 0; j < outLayer.outputSize; j++)
                        {
                            // 列优先格式：第j个类别，第i个样本
                            logitsFile << ", " << hostLogits[j * batch.size + i];
                        }
                        logitsFile << "\n";
                    }

                    delete[] hostLogits;
                    delete[] hostBatchLabels;
                    logitsFile.close();
                }
            }
            // 计算损失（使用logits，即outputTemp）
            calcCrossEntropyLoss<<<blocksPerGrid, threadsPerBlock>>>(outputTemp, batch.labels, deviceLosses, batch.size,
                                                                     outLayer.outputSize);
            flag |= checkCUDA(cudaGetLastError());

            // 传回主机
            checkCUDA(cudaMemcpy(hostPredictions, inputTemp, batch.size * outLayer.outputSize * sizeof(float),
                                 cudaMemcpyDeviceToHost));
            checkCUDA(cudaMemcpy(hostLosses, deviceLosses, batch.size * sizeof(float), cudaMemcpyDeviceToHost));
            checkCUDA(cudaMemcpy(hostLabels, batch.labels, batch.size * sizeof(float), cudaMemcpyDeviceToHost));

            // 计算准确率和累计损失
            for (int i = 0; i < batch.size; i++)
            {
                // 找到预测类别（最大概率的类别）
                int predictedClass = 0;
                float maxProb = hostPredictions[0 * batchSize + i];
                for (int j = 1; j < outLayer.outputSize; j++)
                {
                    if (hostPredictions[j * batchSize + i] > maxProb)
                    {
                        maxProb = hostPredictions[j * batchSize + i];
                        predictedClass = j;
                    }
                }

                // 获取真实标签
                int trueLabel = static_cast<int>(hostLabels[i]);

                // 统计正确预测
                if (predictedClass == trueLabel)
                {
                    totalCorrect++;
                }

                totalLoss += hostLosses[i];
                totalSamples++;
            }

            // 显示进度
            if ((b + 1) % 10 == 0 || (b + 1) == numBatches)
            {
                float currentAccuracy = 100.0f * totalCorrect / totalSamples;
                float currentLoss = totalLoss / totalSamples;
                printf("测试进度: %d/%d batches, 当前准确率: %.2f%%, 当前损失: %.4f\n", b + 1, numBatches,
                       currentAccuracy, currentLoss);
            }
        }

        // 计算最终结果
        float finalAccuracy = 100.0f * totalCorrect / totalSamples;
        float finalLoss = totalLoss / totalSamples;

        printf("\n测试完成！\n");
        printf("总样本数: %d\n", totalSamples);
        printf("正确预测: %d\n", totalCorrect);
        printf("最终准确率: %.2f%%\n", finalAccuracy);
        printf("平均损失: %.4f\n", finalLoss);

        // 释放内存
        delete[] hostPredictions;
        delete[] hostLosses;
        delete[] hostLabels;
        cudaFree(deviceLosses);

        return (ErrorCode)flag;
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
        DIR* dirHandle = opendir(path.c_str());
        if (dirHandle == nullptr)
            return FILE_NOT_FOUND;

        struct dirent* entry;
        while ((entry = readdir(dirHandle)) != nullptr)
        {
            if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, ".."))
            {
                count++;
            }
        }
        closedir(dirHandle);
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
        DIR* dirHandle = opendir(path.c_str());
        if (dirHandle == nullptr)
            return FILE_NOT_FOUND;

        struct dirent* entry;
        while ((entry = readdir(dirHandle)) != nullptr)
        {
            if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, ".."))
            {
                std::string fileName = path + "/" + entry->d_name;
                int width, height, channel;
                unsigned char* img = stbi_load(fileName.c_str(), &width, &height, &channel, 1);

                // 检查图像是否成功加载
                if (img == nullptr)
                {
                    printf("警告: 无法加载图像文件: %s\n", fileName.c_str());
                    continue; // 跳过这个文件，继续处理下一个
                }

                // 检查图像尺寸是否符合预期（MNIST应该是28x28）
                if (width != 28 || height != 28)
                {
                    printf("警告: 图像尺寸不符合预期 (期望28x28, 实际%d x %d): %s\n", width, height, fileName.c_str());
                    stbi_image_free(img);
                    continue; // 跳过这个文件
                }

                // 按行优先顺序读取像素数据
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        data.push_back(static_cast<float>(img[y * width + x]) * inv);
                    }
                }
                labels.push_back(static_cast<float>(type));
                stbi_image_free(img);
                counter++;
            }
        }
        printf("\r%d / %llu", counter, cap);
        closedir(dirHandle);
    }
    printf("\n");

    // 应用数据标准化（使用MNIST的标准值，与PyTorch保持一致）
    // PyTorch: transforms.Normalize((0.1307,), (0.3081,))
    // 标准化公式: (x - mean) / std
    // 注意：数据已经归一化到[0,1]，现在需要标准化
    const float mnist_mean = 0.1307f;
    const float mnist_std = 0.3081f;
    printf("应用数据标准化: (x - %.4f) / %.4f\n", mnist_mean, mnist_std);
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] = (data[i] - mnist_mean) / mnist_std;
    }

    // 打乱数据
    std::vector<int> indices(labels.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));
    std::vector<float> shuffledData(data.size()), shuffledLabels(labels.size());
    for (size_t i = 0; i < indices.size(); i++)
    {
        std::copy(data.begin() + indices[i] * dest->stride, data.begin() + (indices[i] + 1) * dest->stride,
                  shuffledData.begin() + i * dest->stride);
        shuffledLabels[i] = labels[indices[i]];
    }
    // 统计混乱程度（相邻标签不同的比例）
    int diffCount = 0;
    for (size_t i = 1; i < shuffledLabels.size(); i++)
    {
        if (shuffledLabels[i] != shuffledLabels[i - 1])
            diffCount++;
    }
    printf("混乱程度: %.2f%% (%d/%zu 相邻样本标签不同)\n", 100.0f * diffCount / (shuffledLabels.size() - 1), diffCount,
           shuffledLabels.size() - 1);
    int cudaState = 0;
    cudaState |= checkCUDA(cudaMalloc(&dest->x, shuffledData.size() * sizeof(float)));
    cudaState |= checkCUDA(cudaMalloc(&dest->y, shuffledLabels.size() * sizeof(float)));
    cudaState |= checkCUDA(
        cudaMemcpy(dest->x, shuffledData.data(), shuffledData.size() * sizeof(float), cudaMemcpyHostToDevice));
    cudaState |= checkCUDA(
        cudaMemcpy(dest->y, shuffledLabels.data(), shuffledLabels.size() * sizeof(float), cudaMemcpyHostToDevice));
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
constexpr int BATCH_SIZE = 200;

DataSet trainSet;
DataSet testSet;
MLP model(MODEL_DEPTH, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE);
int main(int argc, char* argv[])
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
    ErrorCode modelTestState = model.test(&testSet);
    printf("check modelTestState:%d\n", modelTestState);

    model.free();

    CUBLAS_CHECK(cublasDestroy(cublasH));
    cudaDeviceReset();

    return 0;
}
