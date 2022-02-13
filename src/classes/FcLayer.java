package classes;

import matricesExceptions.DimensionError;

import java.util.ArrayList;

public class FcLayer implements Layer {
    private int[] inputFlatShape, outputShape;
    private Matrice biases, weights, input, output;
    private boolean isFullInit;

    public FcLayer(int[] outputShape) {
//        this.inputShape = new int[]{- 1, - 1};
//        this.inputFlatShape = new int[]{- 1, - 1};
        this.outputShape = outputShape;
        this.biases = Matrice.vide(1, 1);
        this.weights = Matrice.vide(1, 1);
        this.input = Matrice.vide(1, 1);
        this.output = Matrice.vide(1, 1);
        this.isFullInit = false;

    }

    public FcLayer(Object[] args) {}

    private Matrice reshapeList(ArrayList<Matrice> inputs) {
        /*
        :param inputs: array de n matrices de shape (r, c)
        :return: matrice de shape (n * r * c, 1)
         */

        Matrice output = Matrice.vide(this.inputFlatShape[0], 1);
        int k = 0;

        for (Matrice matrice : inputs) {
            for (int i = 0; i < matrice.getRows(); i++) {
                output.setItem(k, 0, matrice.getItem(i, 0));
                k++;
            }
        }

        return output;
    }

    private void fullInit(ArrayList<Matrice> inputs) {
        int[] inputShape = new int[]{inputs.size(), inputs.get(0).getRows() * inputs.get(0).getColumns()};
        this.inputFlatShape = new int[]{inputs.size() * inputs.get(0).getRows() * inputs.get(0).getColumns(), 1};
        this.biases = Matrice.random(this.outputShape[0], 1, - 1, 1);
        this.weights = Matrice.random(this.outputShape[0], inputShape[0], - 1, 1);
    }

    public ArrayList<Matrice> feedForward(ArrayList<Matrice> inputs) throws DimensionError {
        if (! this.isFullInit) {
            this.fullInit(inputs);
            this.isFullInit = true;
        }

        this.input = this.reshapeList(inputs);
        this.output = this.weights.mul(this.input).add(this.biases);

        ArrayList<Matrice> output = new ArrayList<>();
        output.add(this.output);

        return output;
    }

    public ArrayList<Matrice> backPropagation(ArrayList<Matrice> outputGradients, double learningRate) {
        return null;
    }
}
