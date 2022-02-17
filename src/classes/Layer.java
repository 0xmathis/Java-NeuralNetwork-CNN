package classes;

import matricesExceptions.BadShapeError;
import matricesExceptions.DimensionError;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

public class Layer {

    public String toString() {return null;}

    protected int getId() {return 0;}

    protected void toFile() throws IOException {}

    protected void fromFile() throws FileNotFoundException {}

    protected ArrayList<Matrice> feedForward(ArrayList<Matrice> inputs) throws BadShapeError, DimensionError {return null;}

    protected ArrayList<Matrice> backPropagation(ArrayList<Matrice> outputGradients, double learningRate) throws BadShapeError, DimensionError {return null;}
}
