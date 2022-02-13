package classes;

import matricesExceptions.DimensionError;

import java.util.ArrayList;
import java.util.Objects;
import java.util.function.Function;

public class LossLayer implements Layer {
    public static final String BCE = "bce";
    public static final String MSE = "mse";

    private final Function<Matrice[], Double> loss;
    private final Function<Matrice[], Matrice> lossPrime;

    public LossLayer(String typeLoss) {
        if (! Objects.equals(typeLoss, BCE) && ! Objects.equals(typeLoss, MSE)) {
            throw new IllegalStateException("Unexpected value: " + typeLoss);
        }

        if (BCE.equals(typeLoss)) {
            this.loss = LossLayer::BCE;
            this.lossPrime = LossLayer::BCEprime;
        } else {
            this.loss = LossLayer::MSE;
            this.lossPrime = LossLayer::MSEprime;
        }
    }

    public LossLayer(Object[] args) {
        if (! Objects.equals(args[0], BCE) && ! Objects.equals(args[0], MSE)) {
            throw new IllegalStateException("Unexpected value: " + args[0]);
        }

        if (BCE.equals(args[0])) {
            this.loss = LossLayer::BCE;
            this.lossPrime = LossLayer::BCEprime;
        } else {
            this.loss = LossLayer::MSE;
            this.lossPrime = LossLayer::MSEprime;
        }
    }

    private static double BCE(Matrice[] data) { // sum(targets.log(outputs) + (1-targets).log(1-output)) / targets.rows
        Matrice outputs = data[0];
        Matrice targets = data[1];
        try {
            return (targets.hp(outputs.map(Math::log)).add(targets.rsub(1.).hp(outputs.rsub(1.).map(Math::log)))).sum() / targets.getRows();
        } catch (DimensionError e) {
            System.exit(0);
        }
        return 0;
    }

    private static Matrice BCEprime(Matrice[] data) {  // ((1 - targets) / (1 - outputs) - targets / outputs) / targets.rows
        Matrice outputs = data[0];
        Matrice targets = data[1];
        try {
            return (targets.rsub(1.).div(outputs.rsub(1.)).sub(targets.div(outputs))).div((double) targets.getRows());
        } catch (DimensionError e) {
            System.exit(0);
        }
        return null;
    }

    private static double MSE(Matrice[] data) { // sum((targets - outputs)²) / targets.rows
        Matrice outputs = data[0];
        Matrice targets = data[1];
        try {
            return (targets.sub(outputs)).ps(0, 0) / targets.getRows();
        } catch (DimensionError e) {
            System.exit(0);
        }
        return 0;
    }

    private static Matrice MSEprime(Matrice[] data) {  // (outputs - targets) * 2 / targets.rows
        Matrice outputs = data[0];
        Matrice targets = data[1];
        try {
            return (outputs.sub(targets)).mul((double) (2 / targets.getRows()));
        } catch (DimensionError e) {
            System.exit(0);
        }
        return null;
    }

    public double getError(ArrayList<Matrice> outputs, Matrice targets) {
        return this.loss.apply(new Matrice[]{outputs.get(0), targets});  // outputs ne contient que 1 valeur
    }

    public ArrayList<Matrice> getGradient(Matrice outputs, Matrice targets) {
        ArrayList<Matrice> output = new ArrayList<>();
        output.add(this.lossPrime.apply(new Matrice[]{outputs, targets}));
        return output;
    }

    // ne pas enlever, pour que CNN fonctionne
    public ArrayList<Matrice> feedForward(ArrayList<Matrice> inputs) {
        return null;
    }

    public ArrayList<Matrice> backPropagation(ArrayList<Matrice> outputGradients, double learningRate) {
        return null;
    }
}
