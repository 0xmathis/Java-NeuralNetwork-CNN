import classes.CNN;
import classes.Matrice;
import matricesExceptions.BadShapeError;
import matricesExceptions.DimensionError;

import java.io.IOException;

//11_800
public class Main {
    public static void main(String[] args) throws DimensionError, BadShapeError, IOException {
//        Matrice matrice = Matrice.random(180, 166, - 5, 5);

        CNN network = new CNN(0.5);

        network.addLayer(CNN.CONV, new Object[]{5, 6});
        network.addLayer(CNN.ReLU, new Object[]{"max"});
        network.addLayer(CNN.POOL, new Object[]{"max", 4});

        network.addLayer(CNN.CONV, new Object[]{5, 16});
        network.addLayer(CNN.ReLU, new Object[]{"max"});
        network.addLayer(CNN.POOL, new Object[]{"max", 4});

        network.addLayer(CNN.FLAT, new Object[]{});

        network.addLayer(CNN.FC, new Object[]{new int[]{100, 1}});
        network.addLayer(CNN.ReLU, new Object[]{"sigmoid"});

        network.addLayer(CNN.FC, new Object[]{new int[]{2, 1}});
        network.addLayer(CNN.ReLU, new Object[]{"sigmoid"});

        network.addLayer(CNN.LOSS, new Object[]{"mse"});

        train(network, 5, 5);
    }

    public static void train(CNN network, int nbIteration, int frequence) throws DimensionError, BadShapeError, IOException {
        for (int i = 0; i < nbIteration; i++) {
            Matrice input = Matrice.random(180, 166, -5, 5);
            Matrice target = Matrice.random(2, 1, -3, 3);

            network.trainFromExternalData(input, target, i + 1, frequence);
        }
    }
}
