import classes.CNN;
import classes.Matrice;
import matricesExceptions.BadShapeError;
import matricesExceptions.DimensionError;

import java.util.ArrayList;

//11_800
public class Main {
    public static void main(String[] args) throws DimensionError, BadShapeError {
        Matrice matrice = Matrice.random(180, 166, - 5, 5);

        CNN network = new CNN(0.5);

        network.addLayer(CNN.CONV, new Object[]{5, 6});
        network.addLayer(CNN.ReLU, new Object[]{"max"});
        network.addLayer(CNN.POOL, new Object[]{"max", 4});

        network.addLayer(CNN.CONV, new Object[]{5, 16});
        network.addLayer(CNN.ReLU, new Object[]{"max"});
        network.addLayer(CNN.POOL, new Object[]{"max", 4});

        network.addLayer(CNN.FLAT, new Object[]{});

        network.addLayer(CNN.FC, new Object[]{new int[]{20, 1}});
        network.addLayer(CNN.ReLU, new Object[]{"sigmoid"});

        network.addLayer(CNN.FC, new Object[]{new int[]{2, 1}});
        network.addLayer(CNN.ReLU, new Object[]{"sigmoid"});

        network.addLayer(CNN.LOSS, new Object[]{"bce"});

        long start = System.currentTimeMillis();
        ArrayList<Matrice> output = network.feedForward(matrice);
        System.out.println(output.size());
//        System.out.println(Arrays.toString(output.get(0).getShape()));
        network.backPropagation(output.get(0), Matrice.random(2, 1, - 3, 3));
        System.out.println(System.currentTimeMillis() - start);

    }
}
