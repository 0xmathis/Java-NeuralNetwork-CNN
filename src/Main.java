import classes.CNN;
import classes.Matrice;
import matricesExceptions.BadShapeError;
import matricesExceptions.DimensionError;

import java.util.ArrayList;

public class Main {
    public static void main(String[] args) throws DimensionError, BadShapeError {
        Matrice matrice = Matrice.random(15, 15, - 5, 5);

        CNN network = new CNN(0.5);

        network.addLayer(CNN.CONV, new Object[]{3, 5});
        network.addLayer(CNN.ReLU, new Object[]{"max"});
        network.addLayer(CNN.POOL, new Object[]{"max", 3});
        network.addLayer(CNN.FLAT, new Object[]{});
        network.addLayer(CNN.FC, new Object[]{new int[]{5, 1}});
        network.addLayer(CNN.LOSS, new Object[]{"bce"});

        ArrayList<Matrice> input = new ArrayList<>();
        input.add(matrice);
        ArrayList<Matrice> output = network.feedForward(input);
        System.out.println(output.get(0));
        network.backPropagation(output.get(0), Matrice.random(5, 1, - 3, 3));

    }
}
