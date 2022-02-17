import classes.CNN;
import classes.Matrice;
import matricesExceptions.BadShapeError;
import matricesExceptions.DimensionError;

import java.io.IOException;

//11_800
public class Main {
    public static void main(String[] args) throws DimensionError, BadShapeError, IOException {
        Matrice matrice = Matrice.random(180, 166, - 5, 5);

        CNN network = new CNN(0.5);

//        network.addLayer(CNN.CONV, new Object[]{5, 6});
//        network.addLayer(CNN.ReLU, new Object[]{"max"});
//        network.addLayer(CNN.POOL, new Object[]{"max", 4});
//
//        network.addLayer(CNN.CONV, new Object[]{5, 16});
//        network.addLayer(CNN.ReLU, new Object[]{"max"});
//        network.addLayer(CNN.POOL, new Object[]{"max", 4});
//
//        network.addLayer(CNN.FLAT, new Object[]{});
//
//        network.addLayer(CNN.FC, new Object[]{new int[]{100, 1}});
//        network.addLayer(CNN.ReLU, new Object[]{"sigmoid"});
//
//        network.addLayer(CNN.FC, new Object[]{new int[]{2, 1}});
//        network.addLayer(CNN.ReLU, new Object[]{"sigmoid"});
//
//        network.addLayer(CNN.LOSS, new Object[]{"bce"});

        network.addLayer(CNN.CONV, new Object[]{2, 2});
        network.addLayer(CNN.FC, new Object[]{new int[]{2, 1}});
        network.addLayer(CNN.LOSS, new Object[]{"bce"});

        network.feedForward(Matrice.random(5, 5, - 1, 1));

//        train(network, 1);

        System.out.println(train(network, 5));
    }

    public static long train(CNN network, int nbIteration) throws DimensionError, BadShapeError, IOException {
        long time = 0;
        for (int i = 0; i < nbIteration; i++) {
//            Matrice input = Matrice.random(180, 166, - 5, 5);
//            Matrice target = Matrice.random(2, 1, - 3, 3);
            Matrice input = Matrice.random(10, 10, - 5, 5);
            Matrice target = Matrice.random(2, 1, - 3, 3);

            time += network.trainFromExternalData(input, target, i + 1, 5);
        }

        return time / nbIteration;
    }
}
