package classes;

import matricesExceptions.BadShapeError;
import matricesExceptions.DimensionError;

import java.io.IOException;
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

    public final ArrayList<Layer> network;
    private final double learningRate;
    private double error = 0;

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

    public static String from_millisecondes(long milli) {  // a simplifier avec des boucles
        String end = "";

        int heure = (int) (milli / (3.6 * Math.pow(10, 6)));
        int minute = (int) ((milli % (3.6 * Math.pow(10, 6)) / 60_000));
        int seconde = (int) ((milli % 3_600_000 % 60_000) / 1000);
        int milliseconde = (int) (milli % 1000);

        if (heure != 0) {
            end += String.format("%sh", heure);
        }
        if (minute != 0) {
            end += String.format("%sm", minute);
        }
        if (seconde != 0) {
            end += String.format("%ss", seconde);
        }
        end += String.format("%sms", milliseconde);

        return end;
    }

    public static int randint(int max) {  // nombre aléatoire entier dans l'intervalle [| 0, max |]
        Random rand = new Random();

        return rand.nextInt(max + 1);
    }

    public void addLayer(String layer, Object[] args) throws IllegalStateException {
        if (!LAYERS.containsKey(layer)) {
            throw new IllegalStateException("Unexpected value: " + layer);
        }

        this.network.add(LAYERS.get(layer).apply(addToArray(args, this.network.size())));
    }

    public Matrice feedForward(Matrice input) throws DimensionError, BadShapeError, IOException {
        ArrayList<Matrice> data = new ArrayList<>();
        data.add(input);

        // pas de feedForward pour le dernier layer qui sera forcement un lossLayer
        for (Layer layer : this.network.subList(0, this.network.size() - 1)) {
            data = layer.feedForward(data);
        }

        return data.get(0);
    }

    public void backPropagation(Matrice outputs, Matrice targets) throws DimensionError, BadShapeError {
        ArrayList<Matrice> gradient = ((LossLayer) this.network.get(this.network.size() - 1)).getGradient(outputs, targets);
        this.error = ((LossLayer) this.network.get(this.network.size() - 1)).getError(outputs, targets);

        for (Layer layer : reverse(this.network.subList(0, this.network.size() - 1))) {
            gradient = layer.backPropagation(gradient, this.learningRate);
        }
    }

    public void trainFromExternalData(Matrice input, Matrice target, int iteration, int frequence) throws DimensionError, BadShapeError, IOException {
        long start = System.currentTimeMillis();

        Matrice output = this.feedForward(input);
        this.backPropagation(output, target);

        if (iteration % frequence == 0) {
            this.toFile();
        }

        System.out.printf("%s: %s\n", iteration, from_millisecondes(System.currentTimeMillis() - start));
    }

    public double getError() {
        return this.error;
    }

    private Object[] addToArray(Object[] array, int element) {
        Object[] output = new Object[array.length + 1];
        System.arraycopy(array, 0, output, 0, array.length);
        output[array.length] = element;

        return output;
    }

    public void fromFile(int[] inputShape) throws DimensionError, BadShapeError, IOException {
        this.feedForward(Matrice.vide(inputShape[0], inputShape[1]));

        for (Layer layer : this.network) {
            layer.fromFile();
        }
    }

    private void toFile() throws IOException {
        for (Layer layer : this.network) {
            layer.toFile();
        }
    }
}



