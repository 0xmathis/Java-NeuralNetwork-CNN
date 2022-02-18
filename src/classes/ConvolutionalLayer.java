package classes;

import interfaces.Layer;
import matricesExceptions.DimensionError;

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

public class ConvolutionalLayer implements Layer {
    private final int kernelDim, nbKernel;
    private final File valueFile;
    private int inputDepth;
    private int[] inputShape, outputShape;
    private boolean isFullInit;
    private ArrayList<ArrayList<Matrice>> kernels;
    private ArrayList<Matrice> inputs, biases, outputs;

    public ConvolutionalLayer(Object[] args) {
        this.inputShape = new int[]{-1, -1};
        this.outputShape = new int[]{-1, -1};
        this.inputDepth = -1;
        this.kernelDim = (int) args[0];
        this.nbKernel = (int) args[1];
        this.isFullInit = false;
        int id = (int) args[2];

        this.valueFile = new File(String.format("CNN\\CONV%s.txt", id));

        this.kernels = new ArrayList<>();
        this.inputs = new ArrayList<>();
        this.biases = new ArrayList<>();
        this.outputs = new ArrayList<>();
    }

    private static double sum(Matrice matrice) {
        float sum = 0;

        for (int i = 0; i < matrice.getRows(); i++) {
            for (int j = 0; j < matrice.getColumns(); j++) {
                sum += matrice.getItem(i, j);
            }
        }

        return sum;
    }

    private static Matrice rotation180Matrice(Matrice matrice) throws DimensionError {
        ArrayList<ArrayList<Double>> rotArray = new ArrayList<>();

        for (int i = 0; i < matrice.getRows(); i++) {
            ArrayList<Double> subRotMatrice = new ArrayList<>();
            for (int j = 0; j < matrice.getColumns(); j++) {
                if (i + j == matrice.getRows() - 1) {
                    subRotMatrice.add(1.);
                } else {
                    subRotMatrice.add(0.);
                }
            }
            rotArray.add(subRotMatrice);
        }

        Matrice rotMatrice = new Matrice(rotArray);

        return rotMatrice.mul(matrice).mul(rotMatrice);
    }

    private static Matrice validCorrelation(Matrice input, Matrice kernel) throws DimensionError {
        Matrice output = Matrice.vide(input.getRows() - kernel.getRows() + 1, input.getColumns() - kernel.getColumns() + 1);

        for (int i = 0; i < output.getRows(); i++) {
            for (int j = 0; j < output.getColumns(); j++) {
                Matrice subInput = input.getSubMatrice(new int[]{i, j}, new int[]{i + kernel.getRows() - 1, j + kernel.getColumns() - 1});
                output.setItem(i, j, sum(subInput.hp(kernel)));
            }
        }

        return output;
    }

    private static Matrice fullCorrelation(Matrice input, Matrice kernel) throws DimensionError {
        Matrice output = Matrice.vide(input.getRows() + kernel.getRows() - 1, input.getColumns() + kernel.getColumns() - 1);
        Matrice inputExpand = input.copy();

        for (int i = 0; i < kernel.getRows() - 1; i++) {
            inputExpand = inputExpand.addNullRow(Matrice.TOP);
            inputExpand = inputExpand.addNullRow(Matrice.BOTTOM);
            inputExpand = inputExpand.addNullColumn(Matrice.LEFT);
            inputExpand = inputExpand.addNullColumn(Matrice.RIGHT);
        }

        for (int i = 0; i < output.getRows(); i++) {
            for (int j = 0; j < output.getColumns(); j++) {
                Matrice subInput = inputExpand.getSubMatrice(new int[]{i, j}, new int[]{i + kernel.getRows() - 1, j + kernel.getColumns() - 1});
                output.setItem(i, j, sum(subInput.hp(kernel)));
            }
        }
        return output;
    }

    @SuppressWarnings("ResultOfMethodCallIgnored")
    public void toFile() throws IOException {
        this.valueFile.createNewFile();
        Writer writer = new FileWriter(this.valueFile);

        for (Matrice matrice : this.biases) {
            for (int i = 0; i < matrice.getRows(); i++) {
                for (int j = 0; j < matrice.getColumns(); j++) {
                    writer.write(String.format("%s\n", matrice.getItem(i, j)));
                }
            }
        }

        for (ArrayList<Matrice> kernelRow : this.kernels) {
            for (Matrice matrice : kernelRow) {
                for (int i = 0; i < matrice.getRows(); i++) {
                    for (int j = 0; j < matrice.getColumns(); j++) {
                        writer.write(String.format("%s\n", matrice.getItem(i, j)));
                    }
                }
            }
        }

        writer.close();
    }

    public void fromFile() throws FileNotFoundException {
        Scanner scanner = new Scanner(this.valueFile);

        ArrayList<Matrice> biasesValue = new ArrayList<>();
        for (int k = 0; k < this.nbKernel; k++) {
            Matrice matrice = Matrice.vide(this.outputShape[0], this.outputShape[1]);
            for (int i = 0; i < matrice.getRows(); i++) {
                for (int j = 0; j < matrice.getColumns(); j++) {
                    Double data = Double.parseDouble(scanner.nextLine());
                    matrice.setItem(i, j, data);
                }
            }
            biasesValue.add(matrice);
        }

        ArrayList<ArrayList<Matrice>> kernelsValue = new ArrayList<>();
        for (int k = 0; k < this.nbKernel; k++) {
            ArrayList<Matrice> subKernels = new ArrayList<>();
            for (int l = 0; l < this.inputDepth; l++) {
                Matrice matrice = Matrice.vide(this.kernelDim, this.kernelDim);
                for (int i = 0; i < matrice.getRows(); i++) {
                    for (int j = 0; j < matrice.getColumns(); j++) {
                        Double data = Double.parseDouble(scanner.nextLine());
                        matrice.setItem(i, j, data);
                    }
                }
                subKernels.add(matrice);
            }
            kernelsValue.add(subKernels);
        }

        this.biases = biasesValue;
        this.kernels = kernelsValue;
        scanner.close();
    }

    public String toString() {
        return String.format("CONV %skernels %sx%s", this.nbKernel, this.kernelDim, this.kernelDim);
    }

    private void initFull(ArrayList<Matrice> inputs) {
        this.inputShape = inputs.get(0).getShape();
        this.inputDepth = inputs.size();
        this.outputShape = new int[]{this.inputShape[0] - this.kernelDim + 1, this.inputShape[1] - this.kernelDim + 1};

        this.biases = new ArrayList<>();
        for (int i = 0; i < this.nbKernel; i++) {
            this.biases.add(Matrice.random(this.outputShape[0], this.outputShape[1], -1, 1));
        }

        this.kernels = new ArrayList<>();
        for (int i = 0; i < this.nbKernel; i++) {
            ArrayList<Matrice> subKernels = new ArrayList<>();
            for (int j = 0; j < this.inputDepth; j++) {
                subKernels.add(Matrice.random(this.kernelDim, this.kernelDim, -1, 1));
            }
            this.kernels.add(subKernels);
        }

    }

    @SuppressWarnings("unchecked")
    public ArrayList<Matrice> feedForward(ArrayList<Matrice> inputs) throws DimensionError {
        if (!this.isFullInit) {
            this.initFull(inputs);
            this.isFullInit = true;
        }

        this.inputs = (ArrayList<Matrice>) inputs.clone();
        this.outputs = (ArrayList<Matrice>) this.biases.clone();

        for (int k = 0; k < this.nbKernel; k++) {
            for (int l = 0; l < this.inputDepth; l++) {
                this.outputs.set(k, this.outputs.get(k).add(validCorrelation(this.inputs.get(l), this.kernels.get(k).get(l))));
            }
        }

        return this.outputs;
    }

    public ArrayList<Matrice> backPropagation(ArrayList<Matrice> outputGradients, double learningRate) throws DimensionError {
        ArrayList<ArrayList<Matrice>> kernelGradient = new ArrayList<>();
        ArrayList<Matrice> inputGradient = new ArrayList<>();

        //initialisation + remplissage kernelGradient
        for (int i = 0; i < this.nbKernel; i++) {
            ArrayList<Matrice> subKernelGradient = new ArrayList<>();
            for (int j = 0; j < this.inputDepth; j++) {
                subKernelGradient.add(validCorrelation(this.inputs.get(j), outputGradients.get(i)));
            }
            kernelGradient.add(subKernelGradient);
        }

        // initialisation inputGradient
        for (int i = 0; i < this.inputDepth; i++) {
            inputGradient.add(Matrice.vide(this.inputShape[0], this.inputShape[1]));
        }

        // remplissage inputGradient
        for (int i = 0; i < this.nbKernel; i++) {
            for (int j = 0; j < this.inputDepth; j++) {
                inputGradient.set(j, inputGradient.get(j).add(fullCorrelation(outputGradients.get(i), rotation180Matrice(this.kernels.get(i).get(j)))));
            }
        }

        // modification des kernels et biais
        for (int i = 0; i < this.nbKernel; i++) {
            for (int j = 0; j < this.inputDepth; j++) {
                this.kernels.get(i).set(j, this.kernels.get(i).get(j).sub(kernelGradient.get(i).get(j).mul(learningRate)));
                this.biases.set(j, this.biases.get(j).sub(outputGradients.get(j).mul(learningRate)));
            }
        }

        return inputGradient;
    }
}
