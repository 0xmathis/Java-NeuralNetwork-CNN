package classes;

import matricesExceptions.DimensionError;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;

public class FcLayer implements Layer {
    private final int[] outputShape;
    private final int id;
    private final File valueFile;
    private int[] inputShape, inputFlatShape;
    private Matrice biases, weights, input, output;
    private boolean isFullInit;

    public FcLayer(int[] outputShape, int id) {
        this.inputShape = new int[]{- 1, - 1};
        this.inputFlatShape = new int[]{- 1, - 1};
        this.outputShape = outputShape;
        this.biases = Matrice.vide(1, 1);
        this.weights = Matrice.vide(1, 1);
        this.input = Matrice.vide(1, 1);
        this.output = Matrice.vide(1, 1);
        this.isFullInit = false;
        this.id = id;

        this.valueFile = new File(String.format("FC%s", this.id));
    }

    public FcLayer(Object[] args) {
        this.inputShape = new int[]{- 1, - 1};
        this.inputFlatShape = new int[]{- 1, - 1};
        this.outputShape = (int[]) args[0];
        this.biases = Matrice.vide(1, 1);
        this.weights = Matrice.vide(1, 1);
        this.input = Matrice.vide(1, 1);
        this.output = Matrice.vide(1, 1);
        this.isFullInit = false;
        this.id = (int) args[1];

        this.valueFile = new File(String.format("FC%s", this.id));
    }

    public int getId() {
        return this.id;
    }

    @SuppressWarnings("ResultOfMethodCallIgnored")
    public void toFile() throws IOException {
        this.valueFile.createNewFile();
        Writer writer = new FileWriter(this.valueFile);

        for (int i = 0; i < this.weights.getRows(); i++) {
            for (int j = 0; j < this.weights.getColumns(); j++) {
                writer.write(String.format("%s\n", this.weights.getItem(i, j)));
            }

        }

        for (int i = 0; i < this.biases.getRows(); i++) {
            for (int j = 0; j < this.biases.getColumns(); j++) {
                writer.write(String.format("%s\n", this.biases.getItem(i, j)));
            }
        }

        writer.close();
    }

    public String toString() {
        return String.format("FC %s outputs", this.outputShape[0]);
    }

    private Matrice reshapeList(ArrayList<Matrice> inputs) {
        /*
        :param inputs: array de n matrices de shape (r, c)
        :return: matrice de shape (n * r * c, 1)
         */

        Matrice output = Matrice.vide(this.inputFlatShape[0], 1);
        int k = 0;

        for (Matrice matrice : inputs) {
            for (int i = 0; i < matrice.getRows(); i++) {
                output.setItem(k, 0, matrice.getItem(i, 0));
                k++;
            }
        }

        return output;
    }

    private ArrayList<Matrice> reshapeMatrice(Matrice input) {
        /*
        :param input_: matrice de shape (n * r * c, 1)
        :return: array de n matrices de shape (r, c)
        */

        ArrayList<Matrice> outputs = new ArrayList<>();

        for (int n = 0; n < this.inputShape[0]; n++) {
            outputs.add(input.getSubMatrice(new int[]{n * this.inputShape[1], 0}, new int[]{(n + 1) * this.inputShape[1] - 1, 0}));
        }

        return outputs;
    }

    private void fullInit(ArrayList<Matrice> inputs) {
        this.inputShape = new int[]{inputs.size(), inputs.get(0).getRows() * inputs.get(0).getColumns()};
        this.inputFlatShape = new int[]{inputs.size() * inputs.get(0).getRows() * inputs.get(0).getColumns(), 1};
        this.biases = Matrice.random(this.outputShape[0], 1, - 1, 1);
        this.weights = Matrice.random(this.outputShape[0], this.inputFlatShape[0], - 1, 1);

        System.out.println();
        System.out.println(this.weights);
        System.out.println(this.biases);
    }

    public ArrayList<Matrice> feedForward(ArrayList<Matrice> inputs) throws DimensionError {
        if (! this.isFullInit) {
            this.fullInit(inputs);
            this.isFullInit = true;
        }

        this.input = this.reshapeList(inputs);
        this.output = this.weights.mul(this.input).add(this.biases);

        ArrayList<Matrice> output = new ArrayList<>();
        output.add(this.output);

        return output;
    }

    public ArrayList<Matrice> backPropagation(ArrayList<Matrice> outputGradients, double learningRate) throws DimensionError {
        Matrice weightsGradient = outputGradients.get(0).mul(this.input.transpose());
        this.biases = this.biases.sub(outputGradients.get(0).mul(learningRate));
        this.weights = this.weights.sub(weightsGradient.mul(learningRate));

        return this.reshapeMatrice(this.weights.transpose().mul(outputGradients.get(0)));
    }
}
