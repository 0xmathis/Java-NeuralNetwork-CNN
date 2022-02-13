package classes;

import matricesExceptions.BadShapeError;
import matricesExceptions.DimensionError;

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
    public static final String LOSS = "loss";
    public final Map<String, Function<Object[], Layer>> LAYERS = new HashMap<>();


    private final ArrayList<Layer> network;

    public CNN() {
        this.network = new ArrayList<>();

        LAYERS.put(CONV, ConvolutionalLayer::new);
        LAYERS.put(POOL, PoolingLayer::new);
        LAYERS.put(ReLU, ReluLayer::new);
        LAYERS.put(FC, FcLayer::new);
        LAYERS.put(FLAT, ReluLayer::new);
        LAYERS.put(LOSS, LossLayer::new);

    }

    public void addLayer(String layer, Object[] args) throws IllegalStateException {
        if (! LAYERS.containsKey(layer)) {
            throw new IllegalStateException("Unexpected value: " + layer);
        }

        this.network.add(LAYERS.get(layer).apply(args));
    }

    public ArrayList<Matrice> feedForward(ArrayList<Matrice> data) throws DimensionError, BadShapeError {
        for (Layer layer : this.network) {
            data = layer.feedForward(data);
        }

        return data;
    }
}



