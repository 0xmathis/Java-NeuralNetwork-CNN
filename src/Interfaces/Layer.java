package Interfaces;

import classes.Matrice;
import matricesExceptions.BadShapeError;
import matricesExceptions.DimensionError;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

public interface Layer {
    String toString();

    int getId();

    void toFile() throws IOException;

    void fromFile() throws FileNotFoundException;

    ArrayList<Matrice> feedForward(ArrayList<Matrice> inputs) throws BadShapeError, DimensionError;

    ArrayList<Matrice> backPropagation(ArrayList<Matrice> outputGradients, double learningRate) throws BadShapeError, DimensionError;
}
