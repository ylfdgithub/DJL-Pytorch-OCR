## Java 实现OCR

### 需求简介

代码部分要求纯java实现，同时运行环境为ARM+LINUX，而OCR在所难免需要一些类似 .so 动态库的额外依赖，所以最终选择通过DJL（Deep Java Library）运行Paddle转Pytorch的模型。

![image](https://user-images.githubusercontent.com/81354516/187064287-a5d011ec-53d9-4d25-b2db-4d1dec9b0839.png)

### 实现部分

> 由于网上关于DJL的资料少之又少，官方文档实话实说和没有一样，要想熟练掌握DJL感觉应该是需要对Python运行一系列深度学习模型有足够的知识储备，但作为一个java程序员，不懂这个应该也算正常叭，下面仅把我在该项目中的使用经验和理解写出来供大家参考学习。

#### Pom.xml依赖

```xml
<!-- 这里按照官方推荐使用它提供的版本管理器 -->
<dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>ai.djl</groupId>
                <artifactId>bom</artifactId>
                <version>0.18.0</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
</dependencyManagement>

<dependencies>
        <dependency>
            <groupId>ai.djl.pytorch</groupId>
            <artifactId>pytorch-engine</artifactId>
            <scope>runtime</scope>
        </dependency>
        <dependency>
            <groupId>ai.djl</groupId>
            <artifactId>api</artifactId>
        </dependency>
        <dependency>
            <groupId>net.java.dev.jna</groupId>
            <artifactId>jna</artifactId>
            <version>5.3.0</version>
        </dependency>
        <dependency>
            <groupId>ai.djl.pytorch</groupId>
            <artifactId>pytorch-native-auto</artifactId>
            <version>1.9.1</version>
            <scope>runtime</scope>
        </dependency>
        <dependency>
            <groupId>ai.djl.pytorch</groupId>
            <artifactId>pytorch-jni</artifactId>
            <version>1.11.0-0.18.0</version>
            <scope>runtime</scope>
        </dependency>
        <dependency>
            <groupId>org.slf4j</groupId>
            <artifactId>slf4j-api</artifactId>
            <version>1.7.26</version>
        </dependency>

        <dependency>
            <groupId>org.slf4j</groupId>
            <artifactId>slf4j-simple</artifactId>
            <version>1.7.26</version>
        </dependency>
        <dependency>
            <groupId>ai.djl.pytorch</groupId>
            <artifactId>pytorch-model-zoo</artifactId>
        </dependency>
    	<!-- 这个依赖和平台有关，目前是Windows -->
        <dependency>
            <groupId>ai.djl.pytorch</groupId>
            <artifactId>pytorch-native-cpu</artifactId>
            <classifier>win-x86_64</classifier>
            <scope>runtime</scope>
            <version>1.11.0</version>
    	</dependency>
    </dependencies>
```

> 下面这个依赖是上文中最后一个依赖的备选项		
>
> ​		<dependency>
> ​            <groupId>ai.djl.pytorch</groupId>
> ​            <artifactId>pytorch-native-cpu-precxx11</artifactId>
> ​            <classifier>linux-aarch64</classifier>
> ​            <scope>runtime</scope>
> ​            <version>1.11.0</version>
> ​         </dependency>
> ​		
>
> 这个是针对arm+linux的依赖，如果还需要其他平台的依赖，可去官网查询文档https://docs.djl.ai/engines/pytorch/pytorch-engine/index.html  	可能需要科学上网，然后这个依赖很大，有几十MB，如果你的Maven没有换国内源可能要下很久很久，推荐没有换源的朋友还是换一下，科学上网可能都没什么效果。
> 		

#### 代码部分

##### 模型准备

![image](https://user-images.githubusercontent.com/81354516/187064320-0d8289c0-2563-4611-a55f-ff2218fb1f62.png)

随便找个位置放一下就行，记得ppocr_keys_v1.txt和rec模型放一个目录下，因为这个字符对照文件也是要加载的，默认是和识别模型相同目录下。

> 文末有项目的Github链接，文件和代码自取即可

##### translator 准备

DJL在加载模型的时候需要指定translator，大概就是进行模型的输入输出的处理，就比如Pytorch模型只认识你Tensor或者NDArray类型的输入，那么Translator就可以对输入进行处理，也可以对模型的输出进行处理。DJL其实是有提供一些预设的Translator的，但是好像大家很少拿Pytorch来搞OCR，所以貌似DJL没有提供OCR相关的Translator，这个问题也不大，Translator是可以自定义的，只需要自己创建一个java类然后实现官方提供的接口之后重写两个处理输入输出的方法即可。但这里面就涉及到对NDArray类型的处理，这个和Python里面那个是一样的，但是本人其实也不是很懂，只知道是多维数组的数据结构，对深度学习进行了优化，模型处理起来效率更高。让我自定义就有点难了，万幸的是这俩模型是Paddle转过来的，而Paddle其实给我们写了对应的Translator，实测是可以直接用的。

![image](https://user-images.githubusercontent.com/81354516/187064330-8de1caf7-863f-4380-8295-5c566dfb545e.png)

###### PpWordDetectionTranslator

这个Translator是给区域识别的模型用的，代码太长我就不贴了，实际上和DJL Paddle的依赖中提供的是一模一样的

![image](https://user-images.githubusercontent.com/81354516/187064336-bcd792e5-6285-4cc1-a235-6142554ee4ac.png)

###### PpWordRecognitionTranslator

这个就是给文字识别模型用的
![image](https://user-images.githubusercontent.com/81354516/187064338-92fd45c4-e1de-4e59-afea-4e19c8a8419e.png)

###### BoundFinder

这个是区域识别额外依赖的一个文件，这里也稍微贴一下，没有细究是干什么的

![image](https://user-images.githubusercontent.com/81354516/187064345-c2f7ffca-2fee-4297-8ddf-d0abf9e4f894.png)
##### 模型初始化以及Predictor生成

```java
//这个写法其实不太好，绝对地址靠谱点
static String DET_PATH = "src/main/java/org/example/models/ch_ptocr_det_infer.pt";
static String REC_PATH = "src/main/java/org/example/models/ch_ptocr_rec_infer.pt";
//下面代码会有些异常需要处理，我就不写进去了
/**
 * DET模型构建
 */
Criteria<Image, DetectedObjects> criteria_det = Criteria.builder()
        .setTypes(Image.class, DetectedObjects.class)
        .optModelPath(Paths.get(DET_PATH))
    	//加载Translator
        .optTranslator(new PpWordDetectionTranslator(new ConcurrentHashMap<String, String>()))
        .build();
ZooModel<Image, DetectedObjects> detectionModel=criteria_det.loadModel();

/**
 * REC模型构建
 */
 Criteria<Image, String> criteria_rec = Criteria.builder()
        .setTypes(Image.class, String.class)
        .optModelPath(Paths.get(REC_PATH))
        .optTranslator(new PpWordRecognitionTranslator())
        .optProgress(new ProgressBar())
     	.build();
ZooModel<Image, String> recognitionModel=criteria_rec.loadModel();
/**
 * 两个Predictor生成
 */
Predictor<Image, DetectedObjects> detector = detectionModel.newPredictor();
Predictor<Image, String> recognizer = recognitionModel.newPredictor();

```

> 官方文档中称Predictor最好不要重复使用，我重复使用中没有遇到啥问题，反正就在这里提醒一下

##### 图像加载

```java
/**
 * 加载图片，直接传递路径即可
 */
Image img = ImageFactory.getInstance().fromFile(Paths.get(path));
```

这里的Image类型是DJL提供的，官方提供了还算丰富的api吧，但是那个save方法不知怎么的，保存不成功，以前我好像用过的，也可以的，有点困惑。还有两个比较常见的api duplicate()和getWrappedImage()，前者大概就是复制个对象，后者我看源码没咋看懂有啥用，就是返回自己好像。

##### 文字区域检测

```java
/**
 * 文字区域检测
 */
DetectedObjects detectedObj = detector.predict(img);
Image newImage = img.duplicate();
newImage.drawBoundingBoxes(detectedObj);
newImage.getWrappedImage();
```

##### 文字识别

```java
/**
 * 获取分割出来的文字区域列表,并识别返回文本
 */
List<DetectedObjects.DetectedObject> boxes = detectedObj.items();
StringBuilder sb = new StringBuilder();
System.out.println(boxes.size());
for (int i = 0; i < boxes.size(); i++) {
    Image subImage = getSubImage(img, boxes.get(i).getBoundingBox());
    subImage.getWrappedImage();
    String predict = recognizer.predict(subImage);
    sb.append(predict);
}
String result = sb.toString();
```

> 这里就是你们按照自己需求来改了，我这里随便拼接了一下

##### 额外的一些函数

```java
public static Image getSubImage(Image img, BoundingBox box) {
    Rectangle rect = box.getBounds();
    double[] extended = extendRect(rect.getX(), rect.getY(), rect.getWidth(), rect.getHeight());
    int width = img.getWidth();
    int height = img.getHeight();
    int[] recovered = {
            (int) (extended[0] * width),
            (int) (extended[1] * height),
            (int) (extended[2] * width),
            (int) (extended[3] * height)
    };
    return img.getSubImage(recovered[0], recovered[1], recovered[2], recovered[3]);
}

public static double[] extendRect(double xmin, double ymin, double width, double height) {
    double centerx = xmin + width / 2;
    double centery = ymin + height / 2;
    if (width > height) {
        width += height * 1.6;
        height *= 2.6;
    } else {
        height += width * 1.6;
        width *= 2.6;
    }
    double newX = centerx - width / 2 < 0 ? 0 : centerx - width / 2;
    double newY = centery - height / 2 < 0 ? 0 : centery - height / 2;
    double newWidth = newX + width > 1 ? 1 - newX : width;
    double newHeight = newY + height > 1 ? 1 - newY : height;
    return new double[] {newX, newY, newWidth, newHeight};
}
public static Image rotateImg(Image image) {
    try (NDManager manager = NDManager.newBaseManager()) {
        NDArray rotated = NDImageUtils.rotate90(image.toNDArray(manager), 1);
        return ImageFactory.getInstance().fromNDArray(rotated);
    }
}
```

这几个函数呢，我也没全搞懂是干什么的，但是第二个在后续优化中可以起作用，extendRect这个函数是用来扩大文本区域识别框的，因为区域识别之后文本框可能没有全部包住文字，会对后续裁剪子图进行识别造成干扰，所以需要扩大文本框，这里面的几个×的参数可以自己调整调整

#### Github链接

感兴趣的朋友可以去github上拿源码，如果对你解决问题有启发或者帮助可以给点star，如果有什么改进的意见和问题也可以联系我交流交流

