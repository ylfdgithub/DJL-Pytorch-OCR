package org.example.translator;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;


/**
 * A {@link Translator} that post-process the {@link NDArray} into {@link String}
 */
public class PpWordRecognitionTranslator implements NoBatchifyTranslator<Image, String> {

    private List<String> table;
    public PpWordRecognitionTranslator() {
    }
    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        try (InputStream is = ctx.getModel().getArtifact("ppocr_keys_v1.txt").openStream()) {
            table = Utils.readLines(is, true);
            table.add(0, "blank");
            table.add("");
        }

    }
    @Override
    public String processOutput(TranslatorContext ctx, NDList list) throws Exception {
        StringBuilder sb = new StringBuilder();
        NDArray tokens = list.singletonOrThrow();
        long[] indices = tokens.get(new long[]{0L}).argMax(1).toLongArray();
//        System.out.println(indices.length);
        for (int i = 0; i < indices.length; i++) {
            if (i!=indices.length-1&&indices[i]==indices[i+1]&&indices[i]!=0) indices[i]=0;
        }
        int lastIdx = 0;
        for(int i = 0; i < indices.length; ++i) {
            if (indices[i] > 0L && (i <= 0 || indices[i] != (long)lastIdx)) {
                sb.append((String)this.table.get((int)indices[i]));
            }
        }

        return sb.toString();
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
        NDArray img = input.toNDArray(ctx.getNDManager());
//        System.out.println(input.getWidth());
        int[] hw = this.resize32((double)input.getWidth());
        img = NDImageUtils.resize(img, hw[1], hw[0]);
        img = NDImageUtils.toTensor(img).sub(0.5F).div(0.5F);
        img = img.expandDims(0);
        NDList ndArrays = new NDList(new NDArray[]{img});
//        System.out.println(ndArrays);
        return ndArrays;

    }

    private int[] resize32(double w) {
        int width = (int)Math.max(48.0, w) / 48 * 48;
//        System.out.println(width);
        return new int[]{48, width};
    }
}
