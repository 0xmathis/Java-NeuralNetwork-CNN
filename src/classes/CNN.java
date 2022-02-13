package classes;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

public class CNN {
    public static final String CONV = "conv";
    public static final String POOL = "pool";
    public static final String ReLU = "relu";
    public static final String FC = "fc";
    public static final String FLAT = "flat";
    //    public static final Map<String, Function<Object[], Layer>> LAYERS = new HashMap<>();
//    public static final Map<String, Function<Object[], Constructor<?>>> LAYERS = new HashMap<>();
    public final Map<String, Function<Object[], Object>> LAYERS = new HashMap<>();


    private ArrayList<Layer> network;

    public CNN() {
        this.network = new ArrayList<>();

        LAYERS.put(CONV, ConvolutionalLayer::new);
        LAYERS.put(POOL, ReluLayer::new);
        LAYERS.put(ReLU, ReluLayer::new);
        LAYERS.put(FC, ReluLayer::new);
        LAYERS.put(FLAT, ReluLayer::new);

    }

    public void addLayer(String layer, Object[] args) {

    }
}



