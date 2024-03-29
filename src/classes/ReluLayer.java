package classes;

import interfaces.Layer;
import matricesExceptions.DimensionError;

import java.util.ArrayList;
import java.util.Objects;
import java.util.function.Function;

public class ReluLayer implements Layer {
    public static final String MAX = "max";
    public static final String SIGMOID = "sigmoid";
    public static final String TANH = "tanh";
    public static final String STEP = "step";

    String typeReLU;
    ArrayList<Matrice> inputs, outputs;
    Function<Double, Double> activation, activationPrime;

    protected ReluLayer(Object[] args) {
        if (!Objects.equals(args[0], MAX) && !Objects.equals(args[0], SIGMOID) && !Objects.equals(args[0], TANH) && !Objects.equals(args[0], STEP)) {
            throw new IllegalStateException("Unexpected value: " + args[0]);
        }

        this.typeReLU = (String) args[0];
        this.inputs = new ArrayList<>();
        this.outputs = new ArrayList<>();

        switch (this.typeReLU) {
            case MAX -> { // MAX function
                this.activation = (Double x) -> Math.max(0, x);
                this.activationPrime = (Double y) -> {
                    if (y > 0) {
                        return 1.;
                    } else {
                        return 0.;
                    }
                };
            }
            case SIGMOID -> { // SIGMOID finction
                this.activation = (Double x) -> 1 / (1 + Math.exp(-x));
                this.activationPrime = (Double y) -> this.activation.apply(y) * (1 - this.activation.apply(y));
            }
            case TANH -> { // TANH function
                this.activation = Math::tanh;
                this.activationPrime = (Double y) -> Math.pow(1 / Math.cosh(y), 2);
            }
            default -> { // STEP function
                this.activation = (Double x) -> {
                    if (x > 0) {
                        return 1.;
                    } else {
                        return 0.;
                    }
                };
                this.activationPrime = (Double y) -> 0.;
            }
        }
    }

    public void toFile() {

    }

    public void fromFile() {
    }

    @Override
    public String toString() {
        return String.format("ReLU %s", this.typeReLU);
    }

    @SuppressWarnings("unchecked")
    public ArrayList<Matrice> feedForward(ArrayList<Matrice> inputs) {
        this.inputs = (ArrayList<Matrice>) inputs.clone();
        this.outputs = new ArrayList<>();

        for (Matrice matrice : this.inputs) {
            this.outputs.add(matrice.map(this.activation));
        }

        return this.outputs;
    }

    public ArrayList<Matrice> backPropagation(ArrayList<Matrice> outputGradients, double learningRate) throws DimensionError {

        ArrayList<Matrice> inputGradient = new ArrayList<>();
        for (int i = 0; i < this.inputs.size(); i++) {
            inputGradient.add(this.inputs.get(i).map(this.activationPrime).hp(outputGradients.get(i)));
        }

        return inputGradient;
    }
}
