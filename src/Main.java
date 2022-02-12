import classes.Matrice;

import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        ArrayList<ArrayList<Double>> arrayList = new Matrice(new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}).toArrayList();
        System.out.println(arrayList);

        arrayList.get(0).set(0, 5.);

        System.out.println(arrayList);
    }
}
