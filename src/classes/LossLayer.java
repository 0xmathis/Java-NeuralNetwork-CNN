package classes;

import matricesExceptions.BadShapeError;
import matricesExceptions.DimensionError;

import java.util.ArrayList;

public class LossLayer implements Layer {
    public static final String BCE = "bce";
    public static final String MSE = "mse";



    public ArrayList<Matrice> feedForward(ArrayList<Matrice> inputs) throws BadShapeError, DimensionError {
        return null;
    }

    public ArrayList<Matrice> backPropagation(ArrayList<Matrice> outputGradients, double learningRate) throws BadShapeError, DimensionError {
        return null;
    }
}
