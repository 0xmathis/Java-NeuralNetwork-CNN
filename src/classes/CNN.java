package classes;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

public class CNN {
    public static final String CONV = "conv";
    public static final String POOL = "pool";
    public static final String ReLU = "relu";
    public static final String FC = "fc";
    public static final String FLAT = "flat";
    public static final Map<String, Function<String, Layer>> LAYERS = new HashMap<>();

    static {
        LAYERS.put(ReLU, (s -> new ReluLayer(ReluLayer.MAX)));
    }


}
