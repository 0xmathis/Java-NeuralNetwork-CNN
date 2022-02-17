import classes.CNN;
import classes.ImageData;
import classes.Matrice;
import matricesExceptions.BadShapeError;
import matricesExceptions.DimensionError;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

//11_800
public class Main {
    public static String pathCMFD = "Z:\\_DataSets\\TIPE\\CMFD\\";
    public static String pathIMFD = "Z:\\_DataSets\\TIPE\\IMFD\\";
    public static int rowImage = 180;
    public static int columnImage = 166;
    public static int[] dim = new int[]{rowImage, columnImage};
    public static File errorsFile = new File("CNN_errors.txt");

    public static void main(String[] args) throws DimensionError, BadShapeError, IOException {
        String[] listFiles = getListFiles();
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

        trainforTime(network, listFiles, 15 * 60, 1_000);
    }

    public static void trainforTime(CNN network, String[] listFiles, int secondes, int frequence) throws DimensionError, BadShapeError, IOException {
        FileWriter writer = new FileWriter(errorsFile);

        System.out.println("Start");
        long start = System.currentTimeMillis();
        int i = 0;
        while ((System.currentTimeMillis() - start) / 1_000 < secondes) {
            int choix = CNN.randint(listFiles.length - 1);
            ImageData imageData = new ImageData(listFiles[choix], new int[]{rowImage, columnImage});
            Matrice input = imageData.getMatrice();
            Matrice target = imageData.getAnswer().transpose();

            network.trainFromExternalData(input, target, i + 1, frequence);

            writer.write(String.format("%s\n", network.getError()));

            i++;
        }

        System.out.printf("Temps total: %s\n", CNN.from_millisecondes(System.currentTimeMillis() - start));
        writer.close();

        { // pour sauvegarder à la fin
            int choix = CNN.randint(listFiles.length - 1);
            ImageData imageData = new ImageData(listFiles[choix], new int[]{rowImage, columnImage});
            Matrice input = imageData.getMatrice();
            Matrice target = imageData.getAnswer().transpose();

            network.trainFromExternalData(input, target, 1, 1);
        }
    }

    public static String[] getListFiles() {
        String[] repertoireCMFD = new File(pathCMFD).list();
        String[] repertoireIMFD = new File(pathIMFD).list();

        assert repertoireCMFD != null;
        assert repertoireIMFD != null;
        String[] repertoire = new String[repertoireCMFD.length + repertoireIMFD.length];

        for (int i = 0; i < repertoireCMFD.length; i++) {
            repertoire[i] = pathCMFD + repertoireCMFD[i];
        }

        for (int i = 0; i < repertoireIMFD.length; i++) {
            repertoire[repertoireCMFD.length + i] = pathIMFD + repertoireIMFD[i];
        }

        return repertoire;
    }

}
