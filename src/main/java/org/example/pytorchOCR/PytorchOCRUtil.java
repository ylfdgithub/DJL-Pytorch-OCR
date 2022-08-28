package org.example.pytorchOCR;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.example.translator.PpWordDetectionTranslator;
import org.example.translator.PpWordRecognitionTranslator;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

public class PytorchOCRUtil {
    static String DET_PATH = "src/main/java/org/example/models/ch_ptocr_det_infer.pt";
    static String REC_PATH = "src/main/java/org/example/models/ch_ptocr_rec_infer.pt";
    /**
     * DET模型构建
     */
    static Criteria<Image, DetectedObjects> criteria_det = Criteria.builder()
            .setTypes(Image.class, DetectedObjects.class)
            .optModelPath(Paths.get(DET_PATH))
            .optTranslator(new PpWordDetectionTranslator(new ConcurrentHashMap<String, String>()))
            .build();
    static ZooModel<Image, DetectedObjects> detectionModel;

    static {
        try {
            detectionModel = criteria_det.loadModel();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (ModelNotFoundException e) {
            throw new RuntimeException(e);
        } catch (MalformedModelException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * REC模型构建
     */
    static Criteria<Image, String> criteria_rec = Criteria.builder()
            .setTypes(Image.class, String.class)
            .optModelPath(Paths.get(REC_PATH))
            .optTranslator(new PpWordRecognitionTranslator())
            .optProgress(new ProgressBar()).build();
    static ZooModel<Image, String> recognitionModel;

    static {
        try {
            recognitionModel = criteria_rec.loadModel();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (ModelNotFoundException e) {
            throw new RuntimeException(e);
        } catch (MalformedModelException e) {
            throw new RuntimeException(e);
        }
    }


    public static String ocr(String path) throws IOException, TranslateException {
        /**
         * 两个Predictor生成
         */
        Predictor<Image, DetectedObjects> detector = detectionModel.newPredictor();
        Predictor<Image, String> recognizer = recognitionModel.newPredictor();

        /**
         * 加载图片
         */
        Image img = ImageFactory.getInstance().fromFile(Paths.get(path));


        /**
         * 文字区域检测
         */
        DetectedObjects detectedObj = detector.predict(img);
        Image newImage = img.duplicate();
        newImage.drawBoundingBoxes(detectedObj);
        newImage.getWrappedImage();

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
        return sb.toString();
    }







    public PytorchOCRUtil() throws ModelNotFoundException, MalformedModelException, IOException {
    }



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

}
