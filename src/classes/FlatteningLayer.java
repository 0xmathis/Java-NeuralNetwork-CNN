package classes;

import matricesExceptions.BadShapeError;

import java.util.ArrayList;

public class FlatteningLayer implements Layer {
    private int[] inputShape, outputShape;
    private boolean isFullInit;

    public FlatteningLayer() {
        this.inputShape = new int[]{- 1, - 1};
        this.outputShape = new int[]{- 1, - 1};
        this.isFullInit = false;
    }

    public String toString() {
        return "FLATTENING";
    }

    public ArrayList<Matrice> feedForward(ArrayList<Matrice> inputs) throws BadShapeError {
        if (! this.isFullInit) {
            this.inputShape = inputs.get(0).getShape();
            this.outputShape = new int[]{this.inputShape[0] * this.inputShape[1], 1};
            this.isFullInit = true;
        }

        ArrayList<Matrice> outputs = new ArrayList<>();

        for (Matrice matrice : inputs) {
            outputs.add(matrice.reshape(this.outputShape));
        }

        return outputs;
    }

    public ArrayList<Matrice> backPropagation(ArrayList<Matrice> outputGradients, double learningRate) throws BadShapeError {
        ArrayList<Matrice> inputGradients = new ArrayList<>();

        for (Matrice matrice : outputGradients) {
            inputGradients.add(matrice.reshape(this.inputShape));
        }

        return inputGradients;
    }
}
