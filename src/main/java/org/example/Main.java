package org.example;
import java.io.IOException;
import ai.djl.*;
import ai.djl.repository.zoo.*;
import ai.djl.translate.*;


import org.example.pytorchOCR.PytorchOCRUtil;

/**
 * OCR识别Demo
 */
public class Main {
    public static String TEST_IMG_PATH = "src/main/java/org/example/pic/2.jpg";
    public static void main(String[] args) throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
        String ocr = PytorchOCRUtil.ocr(TEST_IMG_PATH);
        System.out.println(ocr);

    }
}