package classes;

import matricesExceptions.BadShapeError;
import matricesExceptions.DimensionError;

import java.util.ArrayList;

public abstract class Layer {

    public abstract String toString();

    public abstract ArrayList<Matrice> feedForward(ArrayList<Matrice> inputs) throws BadShapeError, DimensionError;

    public abstract ArrayList<Matrice> backPropagation(ArrayList<Matrice> outputGradients, double learningRate) throws BadShapeError, DimensionError;
}
