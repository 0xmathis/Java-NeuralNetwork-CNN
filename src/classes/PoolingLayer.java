package classes;

import matricesExceptions.DimensionError;

import java.util.ArrayList;
import java.util.Objects;

public class PoolingLayer implements Layer {
    public static final String MAX = "max";
    public static final String AVG = "average";

    private final String typePooling;
    private final int filterDim, id;
    private int[] inputShape, outputShape;
    private boolean isFullInit;
    private ArrayList<Matrice> inputs, outputs;


    public PoolingLayer(String typePooling, int filterrDim, int id) {
        if (! Objects.equals(typePooling, MAX) && ! Objects.equals(typePooling, AVG)) {
            throw new IllegalStateException("Unexpected value: " + typePooling);
        }

        this.typePooling = typePooling;
        this.filterDim = filterrDim;
        this.inputShape = new int[]{- 1, - 1};
        this.outputShape = new int[]{- 1, - 1};
        this.isFullInit = false;
        this.id = id;

        this.inputs = new ArrayList<>();
        this.outputs = new ArrayList<>();

    }

    public PoolingLayer(Object[] args) {
        if (! Objects.equals(args[0], MAX) && ! Objects.equals(args[0], AVG)) {
            throw new IllegalStateException("Unexpected value: " + args[0]);
        }

        this.typePooling = (String) args[0];
        this.filterDim = (int) args[1];
        this.inputShape = new int[]{- 1, - 1};
        this.outputShape = new int[]{- 1, - 1};
        this.isFullInit = false;
        this.id = (int) args[2];

        this.inputs = new ArrayList<>();
        this.outputs = new ArrayList<>();
    }

    private static double max(Matrice matrice) {
        double currentMax = matrice.getItem(0, 0);

        for (int i = 0; i < matrice.getRows(); i++) {
            for (int j = 0; j < matrice.getColumns(); j++) {
                if (matrice.getItem(i, i) > currentMax) {
                    currentMax = matrice.getItem(i, j);
                }
            }
        }

        return currentMax;
    }

    public static double sum(Matrice matrice) {
        double sum = 0;

        for (int i = 0; i < matrice.getRows(); i++) {
            for (int j = 0; j < matrice.getColumns(); j++) {
                sum += matrice.getItem(i, j);
            }
        }

        return sum;
    }

    public int getId() {
        return this.id;
    }

    public void toFile() {

    }

    public String toString() {
        return String.format("POOL %s %sx%s", this.typePooling, this.filterDim, this.filterDim);
    }

    @SuppressWarnings("unchecked")
    public ArrayList<Matrice> feedForward(ArrayList<Matrice> inputs) {
        if (! this.isFullInit) {
            this.inputShape = inputs.get(0).getShape();
            this.outputShape = new int[]{this.inputShape[0] / filterDim, this.inputShape[1] / this.filterDim};
            this.isFullInit = true;
        }

        this.inputs = (ArrayList<Matrice>) inputs.clone();

        this.outputs = new ArrayList<>();
        for (Matrice matrice : this.inputs) {
            this.outputs.add(pooling(matrice));
        }

        return this.outputs;
    }

    public ArrayList<Matrice> backPropagation(ArrayList<Matrice> outputGradients, double learningRate) throws DimensionError {
        ArrayList<Matrice> inputGradients = new ArrayList<>();

        for (int k = 0; k < outputGradients.size(); k++) {
            inputGradients.add(antiPooling(this.inputs.get(k), this.outputs.get(k).hp(outputGradients.get(k))));
        }

        return inputGradients;
    }

    private Matrice pooling(Matrice input) {
        Matrice output = Matrice.vide(input.getRows() / this.filterDim, input.getColumns() / this.filterDim);

        for (int i = 0; i < output.getRows(); i++) {
            for (int j = 0; j < output.getColumns(); j++) {
                Matrice subInput = input.getSubMatrice(new int[]{i * this.filterDim, j * this.filterDim}, new int[]{(i + 1) * this.filterDim - 1, (j + 1) * this.filterDim - 1});

                if (Objects.equals(this.typePooling, MAX)) {
                    output.setItem(i, j, max(subInput));
                } else {
                    output.setItem(i, j, sum(subInput) / (subInput.getRows() * subInput.getColumns())); // a essayer pour verifier si les sorties sont bien des doubles
                }
            }
        }

        return output;
    }

    private Matrice antiPooling(Matrice input, Matrice output) {
        ArrayList<ArrayList<Double>> antiOutput = new ArrayList<>();

        for (int i = 0; i < this.inputShape[0]; i++) {
            ArrayList<Double> subList = new ArrayList<>();
            for (int j = 0; j < this.inputShape[1]; j++) {
                if (Objects.equals(this.typePooling, MAX)) {
                    if (i / this.filterDim >= output.getRows() || j / this.filterDim >= output.getColumns() || ! Objects.equals(input.getItem(i, j), output.getItem(i / this.filterDim, j / this.filterDim))) {
                        subList.add(0.);
                    } else {
                        subList.add(1.);
                    }
                } else {
                    if (i / this.filterDim >= output.getRows() || j / this.filterDim >= output.getColumns()) {
                        subList.add(0.);
                    } else {
                        subList.add((double) (1 / (this.filterDim * this.filterDim))); // a essayer pour verifier si les sorties sont bien des doubles
                    }
                }
            }
            antiOutput.add(subList);
        }

        return new Matrice(antiOutput);
    }
}
