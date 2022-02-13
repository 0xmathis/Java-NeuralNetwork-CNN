import java.util.ArrayList;
import java.util.Collections;

public class Main {
    public static void main(String[] args) {
//        CNN cnn = new CNN();
//
//        System.out.println(cnn.LAYERS.get(CNN.ReLU).apply(new Object[]{ReluLayer.MAX}));
        ArrayList<Integer> test = new ArrayList<>();

        for (int i = 0; i < 10; i++) {
            test.add(i);
        }

        System.out.println(test);
        Collections.reverse(test);
        System.out.println(test);
        System.out.println(test.get(test.size() - 1));

    }
}
