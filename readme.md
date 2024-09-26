# InternLM 1.8B模型安卓端侧部署实践

书生浦语官网：https://internlm.intern-ai.org.cn/

MindSearch（欢迎 Star）：https://github.com/InternLM/MindSearch



## 笔记

建议先看看部署视频，里面详细介绍了端侧部署的流程，以及实现的原理。

[B站视频链接](https://www.bilibili.com/video/BV1Ai421a7R6/?vd_source=2812aff90e8f21adae9e69e7dbb6f269)

![](.\lx_image\1.png)

这里我就不展开了，直接实践，建议大家一步一步执行哈，了解下每条命令的含义

```python
# 设置国内镜像下载rust
export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
export RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
curl --proto '=https' --tlsv1.2 -sSf https://mirrors.ustc.edu.cn/misc/rustup-install.sh  | sh

# 安装Android Studio
mkdir -p /root/android && cd /root/android
wget https://redirector.gvt1.com/edgedl/android/studio/ide-zips/2024.1.1.12/android-studio-2024.1.1.12-linux.tar.gz
tar -xvzf android-studio-2024.1.1.12-linux.tar.gz
cd android-studio
wget https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip?hl=zh-cn
unzip commandlinetools-linux-11076708_latest.zip\?hl\=zh-cn

# 这一步设置Java_home会提示报错提示无效目录，所以这一步先不执行，我这里删除掉一层Android路径了，前面创建的Android路径，第二天后来就忘记cd进去了，后面全部就没有Android这层目录了。
export JAVA_HOME=/root/android-studio/jbr

cmdline-tools/bin/sdkmanager "ndk;27.0.12077973" "cmake;3.22.1"  "platforms;android-34" "build-tools;33.0.1" --sdk_root='sdk'

# 设置环境变量
. "$HOME/.cargo/env"
export ANDROID_NDK=/root/android-studio/sdk/ndk/27.0.12077973
export TVM_NDK_CC=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang
export JAVA_HOME=/root/android-studio/jbr
export ANDROID_HOME=/root/android-studio/sdk
export PATH=/usr/local/cuda-12/bin:$PATH
export PATH=/root/android-studio/sdk/cmake/3.22.1/bin:$PATH

# 转换模型,安装mlc-llm
conda create --name mlc-prebuilt  python=3.11
conda activate mlc-prebuilt
conda install -c conda-forge git-lfs

# 测试：conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# 详细版本请看PyTorch官网：https://pytorch.org/get-started/previous-versions/
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install transformers sentencepiece protobuf
# 安装mlc
python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu122 mlc-ai-nightly-cu122
#测试下mlc-llm安装是否成功
python -c "import mlc_llm; print(mlc_llm)"
# 克隆项目
git clone https://github.com/mlc-ai/mlc-llm.git
cd mlc-llm
git submodule update --init --recursive
```



**安装mlc_llm方法一**

这里简单介绍上面最后一步转换模型安装mlc-llm需要开发机配置代理，小伙伴们自行配置吧，然后直接执行上面命令下载即可。

**安装mlc_llm方法二**

我是直接本地通过代理访问[https://mlc.ai/wheels](https://mlc.ai/wheels)，然后下载的`mlc_ai_nightly_cu122-0.15.dev559-cp311-cp311-manylinux_2_28_x86_64.whl`和`mlc_llm_nightly_cu122-0.1.dev1520-cp311-cp311-manylinux_2_28_x86_64.whl`，然后上传到开发机里面去的，直接pip安装`python -m pip install mlc_llm_nightly_cu122-0.1.dev1522-cp311-cp311-manylinux_2_28_x86_64.whl mlc_ai_nightly_cu122-0.15.dev559-cp311-cp311-manylinux_2_28_x86_64.whl`命令安装本地whl文件。注意下，这里cuda和python的版本要和我们的环境一致。

![](.\lx_image\2.png)

继续使用 `mlc_llm` 的 `convert_weight` 对模型参数进行转换和量化，转换后的参数可以跨平台使用

```python
mkdir -p /root/models/
ln -s /share/new_models/Shanghai_AI_Laboratory/internlm2_5-1_8b-chat /root/models/internlm2_5-1_8b-chat
cd android/MLCChat  
export TVM_SOURCE_DIR=/root/mlc-llm/3rdparty/tvm
export MLC_LLM_SOURCE_DIR=/root/mlc-llm
# 使用mlc_llm的convert_weight对模型参数进行转换和量化操作
mlc_llm convert_weight /root/models/internlm2_5-1_8b-chat/ \
    --quantization q4f16_1 \
    -o dist/internlm2_5-1_8b-chat-q4f16_1-MLC

# 使用mlc_llm 的 gen_config 生成 mlc-chat-config.json 并处理 tokenizer
mlc_llm gen_config /root/models/internlm2_5-1_8b-chat/      --quantization q4f16_1 --conv-template chatml      -o dist/internlm2_5-1_8b-chat-q4f16_1-MLC

Do you wish to run the custom code? [y/N] y

# 测试转换的模型，这里你用vscode就能看到那个.so文件，注意我们所在的工作路径root/mlc-llm/android/MLCChat#下
mkdir dist/libs
mlc_llm compile ./dist/internlm2_5-1_8b-chat-q4f16_1-MLC/mlc-chat-config.json \
    --device cuda -o dist/libs/internlm2_5-1_8b-chat-q4f16_1-MLC-cuda.so
# 直接编写测试脚本来测试编译后的模型是否符合预期
touch a.py
# a.py
from mlc_llm import MLCEngine

# Create engine
engine = MLCEngine(model="./dist/internlm2_5-1_8b-chat-q4f16_1-MLC", model_lib="./dist/libs/internlm2_5-1_8b-chat-q4f16_1-MLC-cuda.so")

# Run chat completion in OpenAI API.
print(engine)
for response in engine.chat.completions.create(
    messages=[{"role": "user", "content": "你是谁？"}],
    stream=True
):
    for choice in response.choices:
        print(choice.delta.content, end="", flush=True)
print("\n")
engine.terminate()
# 执行脚本
python a.py
# 偷懒解决：如果缺少tokenizer.json或者tokenizer_config.json等其他配置error直接去下面链接下download
https://huggingface.co/timws/internlm2_5-1_8b-chat-q4f16_1-MLC/tree/main
```



上述流程如下截图

![](.\lx_image\3.png)

**记录一个问题**

上面执行使用mlc_llm 的 gen_config 生成 mlc-chat-config.json 并处理 tokenizer的命令时，若出现**`ValueError: The repository for /root/models/internlm2_5-1_8b-chat contains custom code which must be executed to correctly load the model. You can inspect the repository content at https://hf.co//root/models/internlm2_5-1_8b-chat.Please pass the argument trust_remote_code=True to allow custom code to be run.`**

我们去提示的路径`cd /root/.conda/envs/mlc-prebuilt/lib/python3.11/site-packages/mlc_llm/interface/`执行`vim gen_config.py`命令，把该文件调用 `AutoTokenizer` 类加载模型代码段添加如下：

```python
 fast_tokenizer = AutoTokenizer.from_pretrained(str(config.parent), use_fast=True)
```

> ![](.\lx_image\10.png)
>
> 再次执行`mlc_llm gen_config /root/models/internlm2_5-1_8b-chat/      --quantization q4f16_1 --conv-template chatml      -o dist/internlm2_5-1_8b-chat-q4f16_1-MLC`即可正常运行
>
> ![](.\lx_image\9.png)

这里是转换模型生成.so文件的截图

![](.\lx_image\11.png)



执行脚本`python3 a.py`，可能会出现以下报错。据我了解这个是cuda驱动版本的MLC库兼容性的问题，所以我们需要去了解MLC官方文档检查cuda相关信息，这里通过`nvcc --version`明确知道当前的CUDA版本为12.2.

![](.\lx_image\6.png)

记得上面的测试需要测试通过，生成可编译的.so二进制文件截图如下

![](.\lx_image\5.png)

这里简单小结一下，上面环境及mlc库的搭建部署的命令尤为重要，不要遗漏步骤（尤其是设置export环境变量的时候，重启开发机就需要重新设置），不然就会出现error或者其他环境问题，根据报错提示，去解决即可，上述流程在**InternStudio开发机**上并没有成功运行测试的模型，应该是服务器缺乏某些驱动组件原因导致的，出现上述**Aborted**报错，服务器毕竟是docker镜像，有些权限问题，网络问题可能会阻碍流程，强烈建议有本地服务器（**和云服务器类似配置或者Android端侧部署最低配置**）的同学，在本地调试，能够解决大家在开发机上遇到的问题

#### 打包运行

接下来我们准备打包运行了，首先修改`mlc-package-config.json`文件，这个文件的路径我这里是在`/root/mlc-llm/android/MLCChat/`下，这个路径下的`dist目录`就是我们internlm2_5-1_8b-chat-q4f16_1-MLC模型存放位置。

```python
# modelscope下载
modelscope download --model YanfeiSong/Qwen2.5-7B-Instruct-q4f16_1-MLC --local_dir /root/mlc-llm/android/MLCChat/dist
# git 下载
git clone https://www.modelscope.cn/liuchenguangqnm/Phi-3-mini-4k-instruct-q4f16_1-MLC.git

# 修改mlc-package-config.json
{
    "device": "android",
    "model_list": [
        {
            "model": "/root/mlc-llm/android/MLCChat/dist/internlm2_5-1_8b-chat-q4f16_1-MLC",
            "estimated_vram_bytes": 3980990464,
            "model_id": "internlm2_5-1_8b-chat-q4f16_1-MLC",
            "bundle_weight": true
        },
        {
            "model": "/root/mlc-llm/android/MLCChat/gemma-2b-it-q4f16_1-MLC",
            "model_id": "gemma-2b-q4f16_1-MLC",
            "estimated_vram_bytes": 3980990464,
            "bundle_weight": true
        }
    ]
}
# 执行打包命令
mlc_llm package
```



结果如下：

![](.\lx_image\7.png)

然后我们创建签名，上面打包步骤存在报错的情况，建议检查export设置的环境变量的路径以及相关操作步骤是否正确运行。

```python
# 进入指定路径路径
cd /root/mlc-llm/android/MLCChat
# 生成密钥库（使用 keytool 工具生成一个新的密钥库文件 my-release-key.jks，使用 RSA 算法，密钥大小为 2048 位，有效期为 10,000 天）
/root/android-studio/jbr/bin/keytool -genkey -v -keystore my-release-key.jks -keyalg RSA -keysize 2048 -validity 10000

```



#### 修改gradle配置

接下来要修改Android项目中的`build.gradle`文件，该文件路径为`mlc-llm/android/MLCChat/app/build.gradle`，主要就是修改我们的`signingConfigs`签名地址，如下

```python
plugins {
    id 'com.android.application'
    id 'org.jetbrains.kotlin.android'
}

android {
    namespace 'ai.mlc.mlcchat'
    compileSdk 34

    defaultConfig {
        applicationId "ai.mlc.mlcchat"
        minSdk 26
        targetSdk 33
        versionCode 1
        versionName "1.0"

        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary true
        }
    }

    
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = '1.8'
    }
    buildFeatures {
        compose true
    }
    composeOptions {
        kotlinCompilerExtensionVersion '1.4.3'
    }
    packagingOptions {
        resources {
            excludes += '/META-INF/{AL2.0,LGPL2.1}'
        }
    }
# 新增
    signingConfigs {
        release {
            storeFile file("/root/mlc-llm/android/MLCChat/my-release-key.jks")
            storePassword "lx123456"
            keyAlias "mykey"
            keyPassword "lx123456"
        }
    }
# 新增
    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
            // 应用签名
            signingConfig signingConfigs.release
        }
    }
}

dependencies {
    implementation project(":mlc4j")
    implementation 'androidx.core:core-ktx:1.10.1'
    implementation 'androidx.lifecycle:lifecycle-runtime-ktx:2.6.1'
    implementation 'androidx.activity:activity-compose:1.7.1'
    implementation platform('androidx.compose:compose-bom:2022.10.00')
    implementation 'androidx.lifecycle:lifecycle-viewmodel-compose:2.6.1'
    implementation 'androidx.compose.ui:ui'
    implementation 'androidx.compose.ui:ui-graphics'
    implementation 'androidx.compose.ui:ui-tooling-preview'
    implementation 'androidx.compose.material3:material3:1.1.0'
    implementation 'androidx.compose.material:material-icons-extended'
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'androidx.navigation:navigation-compose:2.5.3'
    implementation 'com.google.code.gson:gson:2.10.1'
    implementation fileTree(dir: 'src/main/libs', include: ['*.aar', '*.jar'], exclude: [])
    testImplementation 'junit:junit:4.13.2'
    androidTestImplementation 'androidx.test.ext:junit:1.1.5'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.5.1'
    androidTestImplementation platform('androidx.compose:compose-bom:2022.10.00')
    androidTestImplementation 'androidx.compose.ui:ui-test-junit4'
    debugImplementation 'androidx.compose.ui:ui-tooling'
    debugImplementation 'androidx.compose.ui:ui-test-manifest'

}
```



然后就是命令行编译

```python
cd root/mlc-llm/android/MLCChat
./gradlew assembleRelease
```



![](.\lx_image\12.png)

这个过程很慢，没有配置代理，编译了40min，在安装`app/build/outputs/apk/release`生成`app-release.apk`包,下载到手机上运行运行App需要可以访问huggingface下载模型（参考文档中的bundle方法需要ADB刷入模型数据）

![](.\lx_image\13.png)

### 手机端运行体验

这里需要打开cmd使用adb调试工具来调试，已经将adb工具上传至当前同级目录`mlc_llm package`下，有需要自己本地安装apk。

- 运行应用程序需要能够访问huggingface下载模型
- 4G运行内存
- 如果运行闪退，并且可能是下载不完整可以删除重新下载

这里可能会出现apk无法正常安装及以下问题，

![](.\lx_image\14.png)

先需要执行，安装则连接手机，打开开发者模式，开启允许调试的请求，使用adb来安装

```python
# 使用mlc-chat-config.json规范编译模型库
mlc_llm compile ./dist/internlm2_5-1_8b-chat-q4f16_1-MLC/mlc-chat-config.json --device android -o dist/internlm2_5-1_8b-chat-q4f16_1-MLC/internlm2_5-1_8b-chat-q4f16_1-MLC.tar

cd mlc_llm/cd android/MLCChat
mlc_llm package
./gradlew assembleRelease
# 接下来就是在本地cmd中使用adb了
adb install app-release.apk
adb push internlm2_5-1_8b-chat-q4f16_1-MLC /data/local/tmp/internlm2_5-1_8b-chat-q4f16_1-MLC/
# 创建目录
adb shell "mkdir -p /storage/emulated/0/Android/data/ai.mlc.mlcchat/files/"
# 移动
adb shell "mv /data/local/tmp/internlm2_5-1_8b-chat-q4f16_1-MLC /storage/emulated/0/Android/data/ai.mlc.mlcchat/files/"
# 查看目录信息
adb shell ls -la /storage/emulated/0/Android/data/ai.mlc.mlcchat/files/
```



![](.\lx_image\15.png)

### 总结

这节主要涉及的`internlm2_5-1_8b-chat`模型量化部署到Android端侧，主要还是存在开发机代理下载速度的问题，也可以在本地Android Studio上面去部署实践下，整个过程出现的问题大多数都是依赖性和编译问题，这里采用的是本地已经量化后的Internlm2.5-1.8b的对话模型，感兴趣的小伙伴可以将`mlc-package-config.json`里面的模型都下载到本地试试，也可以自己量化一下，拿到端侧去体验。整体的流程步骤比较快速，希望大家多多理解每一行命令以及代码的含义，期待和大佬们交流互动。