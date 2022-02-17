package classes;

import matricesExceptions.BadShapeError;
import matricesExceptions.DimensionError;

import java.io.IOException;
import java.util.ArrayList;

public interface Layer {

    String toString();

    int getId();

    void toFile() throws IOException;

    ArrayList<Matrice> feedForward(ArrayList<Matrice> inputs) throws BadShapeError, DimensionError;

    ArrayList<Matrice> backPropagation(ArrayList<Matrice> outputGradients, double learningRate) throws BadShapeError, DimensionError;
}
