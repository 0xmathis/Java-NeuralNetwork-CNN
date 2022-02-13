package classes;

import matricesExceptions.BadShapeError;
import matricesExceptions.DimensionError;

import java.util.*;
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
    private final double learningRate;

    public CNN(double learningRate) {
        this.learningRate = learningRate;
        this.network = new ArrayList<>();

        LAYERS.put(CONV, ConvolutionalLayer::new);
        LAYERS.put(POOL, PoolingLayer::new);
        LAYERS.put(ReLU, ReluLayer::new);
        LAYERS.put(FC, FcLayer::new);
        LAYERS.put(FLAT, FlatteningLayer::new);
        LAYERS.put(LOSS, LossLayer::new);

    }

    private static ArrayList<Layer> reverse(List<Layer> arrayList) {
        ArrayList<Layer> output = new ArrayList<>();

        for (Layer layer : arrayList) {
            output.add(0, layer);
        }

        return output;
    }

    public void addLayer(String layer, Object[] args) throws IllegalStateException {
        if (! LAYERS.containsKey(layer)) {
            throw new IllegalStateException("Unexpected value: " + layer);
        }

        this.network.add(LAYERS.get(layer).apply(args));
    }

    public ArrayList<Matrice> feedForward(Matrice input) throws DimensionError, BadShapeError {
        ArrayList<Matrice> data = new ArrayList<>();
        data.add(input);
        System.out.println(Arrays.toString(input.getShape()));

        // pas de feedForward pour le dernier layer qui sera forcement un lossLayer
        for (Layer layer : this.network.subList(0, this.network.size() - 1)) {
            System.out.println(layer);
            data = layer.feedForward(data);
            System.out.println(Arrays.toString(data.get(0).getShape()));
        }

        return data;
    }

    public void backPropagation(Matrice outputs, Matrice targets) throws DimensionError, BadShapeError {
        System.out.println("BackPropagation");

        ArrayList<Matrice> gradient = ((LossLayer) this.network.get(this.network.size() - 1)).getGradient(outputs, targets);


        for (Layer layer : reverse(this.network.subList(0, this.network.size() - 1))) {
            System.out.println(layer);
            gradient = layer.backPropagation(gradient, this.learningRate);
        }
    }
}



