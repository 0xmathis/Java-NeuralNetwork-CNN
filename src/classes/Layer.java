package classes;

import matricesExceptions.BadShapeError;
import matricesExceptions.DimensionError;

import java.util.ArrayList;

public interface Layer {

    String toString();

    ArrayList<Matrice> feedForward(ArrayList<Matrice> inputs) throws BadShapeError, DimensionError;

    ArrayList<Matrice> backPropagation(ArrayList<Matrice> outputGradients, double learningRate) throws BadShapeError, DimensionError;
}
