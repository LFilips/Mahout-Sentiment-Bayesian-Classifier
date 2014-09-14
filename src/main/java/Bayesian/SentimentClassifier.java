package Bayesian;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.naivebayes.ComplementaryNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.training.TrainNaiveBayesJob;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.DocumentProcessor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by luca on 09/09/14.
 */


public class SentimentClassifier {


    static Logger mylog = Logger.getLogger(SentimentClassifier.class);

    public static void main(String args[]) {

        PropertyConfigurator.configure("Logger/log4j.properties");


        //nuovo classificatore che utilizza il bag of word

        String sentimentDictionary = "resources/SentimentDictionary.csv";
        String bagOfWordDirectories = "resources/bagOfWord";


        try {


            createBagOfWords(sentimentDictionary, bagOfWordDirectories);


            String model = "resources/model";
            String trainingSet = "resources/bagOfWord/tf-vectors";
            String labelIndex = "resources/labelindex/labelindex";

            File labelDir = new File("resources/labelindex");

            if (!labelDir.isDirectory())
                labelDir.mkdir();

            Configuration conf = new Configuration();
            FileSystem fs = FileSystem.get(conf);


            TrainNaiveBayesJob classifierTrainer = new TrainNaiveBayesJob();


            mylog.debug("Creating the model");


            String[] option = {
                    "-i", trainingSet, //training set as term frequency vector
                    "-el",
                    "-li", labelIndex, //output folder fol labelindex
                    "-o", model,
                    "-ow",
                    "-c", //complementary naive bayes for umbalanced dataset

            };


            ToolRunner.run(conf, classifierTrainer, option);


            mylog.debug("Inizia la classificazione dei tweet a partire dal modello creato precedentemente");


            NaiveBayesModel Sentimentmodel = NaiveBayesModel.materialize(new Path(model), new Configuration()); //output path of Model
            ComplementaryNaiveBayesClassifier classifier = new ComplementaryNaiveBayesClassifier(Sentimentmodel);


            HashMap<Integer, String> dictionary = loadDictionary(bagOfWordDirectories + "/dictionary.file-0", fs, conf);
            HashMap<String, Integer> reverseDictionary = loadReverseDictionary(bagOfWordDirectories + "/dictionary.file-0", fs, conf);


            HashMap<Integer, String> labelList = loadLabelList(labelIndex);


            analyzeSentiment("io bello intelligente bravo", reverseDictionary, classifier, labelList);
            analyzeSentiment("deprimente terribile", reverseDictionary, classifier, labelList);
            analyzeSentiment("ciao casa sedia", reverseDictionary, classifier, labelList);


        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }


    public static void analyzeSentiment(String text, HashMap<String, Integer> reverseDictionary,
                                        ComplementaryNaiveBayesClassifier classifier, HashMap<Integer, String> labelList) throws IOException, IllegalAccessException, InstantiationException {
        //ho messo void perche stampo in output tutti i risultato che servono

        Vector vector = getVectorFromString(text, reverseDictionary);


        Vector probabilities = classifier.classifyFull(vector);


        mylog.debug("Testo da classificare: " + text + ", Vettore calcolato: " + vector.toString());


        double max = -1;
        int label = 0;
        double actualValue;
        int equalCount = 0;


        for (int k = 0; k < probabilities.size(); k++) {

            actualValue = probabilities.get(k);

//            mylog.debug("Actual Value "+Double.toString(actualValue));
//            mylog.debug("Max Value "+Double.toString(max));
//
//            mylog.debug("Compare: "+(actualValue==max));


            if (actualValue > max) {
                //cambio del massimo
                max = actualValue;
                label = k;


            }


        }


        String sentimentList = "";

        //searching inside the vector all the value that is equal to max
        for (int k = 0; k < probabilities.size(); k++) {

            actualValue = probabilities.get(k);

            if (actualValue == max) {

                sentimentList += labelList.get(k) + " ";


            }


        }


        mylog.debug("Dominant Sentiment  :" + sentimentList + ", probabilities vector: " + probabilities.toString());


    }


    public static HashMap<Integer, String> loadLabelList(String labelIndex) throws IOException {


        HashMap<Integer, String> labelList = new HashMap<Integer, String>();


        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);


        SequenceFile.Reader sfreader = new SequenceFile.Reader(fs, new Path(labelIndex), new Configuration());


        Text key = new Text();
        IntWritable value = new IntWritable();


        // in this way I should create an instance of the correct class


        mylog.debug("Loading label list:");

        while (sfreader.next(key, value)) {


            System.out.println("Label: " + key.toString() + " LabelIndex:"
                    + value.toString());

            labelList.put(value.get(), key.toString());

        }


        mylog.debug("Loaded: " + labelList.size() + " label");


        return labelList;
    }


    /**
     * Create tf vector for the input dataaset
     *
     * @param sentimentDictionary
     * @param outputDirName
     */

    public static void createBagOfWords(String sentimentDictionary, String outputDirName) throws IOException, ClassNotFoundException, InterruptedException {


        Path path = new Path(outputDirName + "/InitialSeqFile/part-m-00000");
        System.out.println("Creating configuration file");
        Configuration conf = new Configuration();


        System.out.println("Creating fs object");
        FileSystem fs = FileSystem.get(conf);

        //creo un sequenceFile con coppia Chiave-Valore long-text (idtweet e contenuto)
        SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, Text.class, Text.class);

        int count = 0;
        BufferedReader reader = new BufferedReader(new FileReader(sentimentDictionary));
        Text key = new Text();
        Text value = new Text();
        while (true) {
            String line = reader.readLine();
            if (line == null) {
                break;
            }

            String[] tokens = line.split(",", 2);
            if (tokens.length != 2) {
                System.out.println("Skip line: " + line);
                System.out.println("Token size" + tokens.length);


                continue;
            }

            String label = tokens[0];

            String word = tokens[1];


            //in base al sorgente,mahout parsa i label facendo split("/");
            key.set("/" + label + "/");
            value.set(word);
            writer.append(key, value);
            count++;


        }
        writer.close();
        System.out.println("Wrote " + count + " entries.");


        int minSupport = 1;


        int maxNGramSize = 3;
        int minLLRValue = 750;


        int reduceTasks = 1;// the number of reducer

        int chunkSize = 64;// the number of chunk stored in each node

        float tfNorm = -1.0f;
        boolean tfLogNormalization = false;


        boolean sequentialAccessOutput = true;


        //here i'm using a simple whitespace analyzer

        Analyzer analyzer = new WhitespaceAnalyzer(Version.LUCENE_48);

        String tokenizedDirectory = outputDirName + DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER;
        Path tokenizedPath = new Path(outputDirName,
                DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);

        System.out.println("Directory tokenizzata: " + tokenizedDirectory);
        if (!new File((outputDirName + "/" + DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER)).exists()) {
            DocumentProcessor.tokenizeDocuments(new Path(outputDirName + "/InitialSeqFile"), // initial
                    // directory
                    // of
                    // vector
                    analyzer.getClass().asSubclass(Analyzer.class), tokenizedPath, // tokenized-document
                    // path
                    conf);
        } else
            System.out.println(
                    "tokenized document directory exist, skipping mapReduce");


        // chiusura dell'analyzer anche se nn so ancora perchè
        analyzer.close();


        // create the tf-vector and wordcount folder


        if (!new File(outputDirName + "/" + DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER).exists()) {
            DictionaryVectorizer.createTermFrequencyVectors(tokenizedPath,
                    new Path(outputDirName),
                    DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER, conf,
                    minSupport, maxNGramSize, minLLRValue, // minimun log-likelyhood
                    // ratio used only when
                    // n-gram > 1
                    tfNorm, // normalization value, if there is tf-idf phase after
                    // have to be -1.0f
                    tfLogNormalization, // normalization boolean,if there is tf-idf
                    // phase after have to be false
                    reduceTasks, chunkSize, sequentialAccessOutput, true); // named
            // Vector
            // da
            // vedere
        } else
            System.out.println("tf file exist");


        return;

    }


    public static Vector getVectorFromString(String text, HashMap<String, Integer> reverseDictionary) {

        Vector vector = new RandomAccessSparseVector(reverseDictionary.size());
        Integer index;

        String token[] = text.split(" ");

        ArrayList<Integer> tokenList = new ArrayList<Integer>();


        for (String e : token) {

            index = reverseDictionary.get(e);

            if (index != null && !(tokenList.contains(index))) {

                vector.set(index.intValue(), 1.0);


            }


        }


        return vector;


    }


    public static HashMap<Integer, String> loadDictionary(String dictionaryFile, FileSystem fs, Configuration conf) throws IOException {

        mylog.debug("Loading dictionary file: " + dictionaryFile);


        HashMap<Integer, String> dictionaryMap = new HashMap<Integer, String>();

        // opening sequenceFIle reader
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(
                dictionaryFile), conf);

        int i = 0; // counter for the print

        Writable key;
        Writable value;
        try {
            // <?> indicates that I'don't know the class to be modeled, perhaps
            // I can put Class<Writable>
            Class<?> keyClass = reader.getKeyClass();
            Class<?> valueClass = reader.getValueClass();

            // in this way I should create an instance of the correct class
            key = (Writable) (keyClass.newInstance());
            value = (Writable) (valueClass.newInstance());

            System.out.println("Key class: " + keyClass.toString()
                    + " Value class: " + valueClass.toString());

            // allocate key-vale based on the writing function

            while (reader.next(key, value)) {
                dictionaryMap.put(Integer.parseInt(value.toString()), key.toString());
                i++;
            }

        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        }
        reader.close();

        mylog.debug("Number of record read from dictionaryfile: " + i);


        return dictionaryMap;

    }

    public static HashMap<String, Integer> loadReverseDictionary(String dictionaryFile, FileSystem fs, Configuration conf) throws IOException {

        mylog.debug("Loading Reverse dictionary file: " + dictionaryFile);


        HashMap<String, Integer> dictionaryMap = new HashMap<String, Integer>();

        // opening sequenceFIle reader
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(
                dictionaryFile), conf);

        int i = 0; // counter for the print

        Writable key;
        Writable value;
        try {
            // <?> indicates that I'don't know the class to be modeled, perhaps
            // I can put Class<Writable>
            Class<?> keyClass = reader.getKeyClass();
            Class<?> valueClass = reader.getValueClass();

            // in this way I should create an instance of the correct class
            key = (Writable) (keyClass.newInstance());
            value = (Writable) (valueClass.newInstance());

            System.out.println("Key class: " + keyClass.toString()
                    + " Value class: " + valueClass.toString());

            // allocate key-vale based on the writing function

            while (reader.next(key, value)) {
                dictionaryMap.put(key.toString(), Integer.parseInt(value.toString()));
                i++;
            }

        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        }
        reader.close();

        mylog.debug("Number of record read from dictionaryfile: " + i);


        return dictionaryMap;

    }


}
