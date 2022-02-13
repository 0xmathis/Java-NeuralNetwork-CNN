import classes.CNN;
import classes.ReluLayer;

import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        CNN cnn = new CNN();

        System.out.println(cnn.LAYERS.get(CNN.ReLU).apply(new Object[]{ReluLayer.MAX}));
    }
}
