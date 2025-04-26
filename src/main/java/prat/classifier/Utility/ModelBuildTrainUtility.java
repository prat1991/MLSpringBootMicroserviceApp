package prat.classifier.Utility;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import jakarta.annotation.PostConstruct;

import java.io.StringReader;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.logging.Level;
import java.util.logging.Logger;

@Component
public class ModelBuildTrainUtility {
    private static final Logger LOGGER = Logger.getLogger(ModelBuildTrainUtility.class.getName());
    
    // Thread-safe lock for model updates
    private final ReadWriteLock rwLock = new ReentrantReadWriteLock();
    
    // Don't autowire these fields - they will be initialized in the methods
    private Instances data;
    private Instances trainData;
    private Instances testData;
    private J48 classifier;
    
    // Model performance metrics
    private double accuracy;
    private String lastTrainingTime;
    
    @PostConstruct
    public void init() {
        try {
            // Initial model training at startup
            refreshModel();
            LOGGER.info("Initial model training completed successfully");
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error during initial model training", e);
            throw new RuntimeException("Failed to initialize model", e);
        }
    }
    
    /**
     * Scheduled retraining of the model to handle data drift
     * Runs every day at 2 AM (cron = "0 0 2 * * ?")
     */
    @Scheduled(cron = "0 0 2 * * ?")
    public void scheduledModelRefresh() {
        try {
            LOGGER.info("Starting scheduled model retraining");
            refreshModel();
            LOGGER.info("Scheduled model retraining completed successfully. New accuracy: " + accuracy);
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error during scheduled model retraining", e);
        }
    }
    
    /**
     * Central method to refresh the model - handles all the steps
     * This is called both at startup and during scheduled refreshes
     */
    private void refreshModel() throws Exception {
        // Write lock - blocks all readers during model update
        rwLock.writeLock().lock();
        try {
            loadDataset();
            preprocessDataset();
            trainModel();
            evaluateModel();
            
            // Update last training timestamp
            lastTrainingTime = java.time.LocalDateTime.now().toString();
        } finally {
            rwLock.writeLock().unlock();
        }
    }
    
    // Step 1: Load the Dataset
    private void loadDataset() throws Exception {
        try {
            // Create a hardcoded in-memory dataset instead of loading from URL
            // In production, this would likely come from a database or file system
            StringBuilder arffBuilder = new StringBuilder();
            arffBuilder.append("@relation iris\n\n");
            arffBuilder.append("@attribute sepallength numeric\n");
            arffBuilder.append("@attribute sepalwidth numeric\n");
            arffBuilder.append("@attribute petallength numeric\n");
            arffBuilder.append("@attribute petalwidth numeric\n");
            arffBuilder.append("@attribute class {Iris-setosa,Iris-versicolor,Iris-virginica}\n\n");
            arffBuilder.append("@data\n");
            arffBuilder.append("5.1,3.5,1.4,0.2,Iris-setosa\n");
            arffBuilder.append("4.9,3.0,1.4,0.2,Iris-setosa\n");
            arffBuilder.append("4.7,3.2,1.3,0.2,Iris-setosa\n");
            arffBuilder.append("4.6,3.1,1.5,0.2,Iris-setosa\n");
            arffBuilder.append("5.0,3.6,1.4,0.2,Iris-setosa\n");
            arffBuilder.append("7.0,3.2,4.7,1.4,Iris-versicolor\n");
            arffBuilder.append("6.4,3.2,4.5,1.5,Iris-versicolor\n");
            arffBuilder.append("6.9,3.1,4.9,1.5,Iris-versicolor\n");
            arffBuilder.append("6.3,3.3,6.0,2.5,Iris-virginica\n");
            arffBuilder.append("5.8,2.7,5.1,1.9,Iris-virginica\n");
            
            // Use StringReader to load the ARFF data
            StringReader reader = new StringReader(arffBuilder.toString());
            
            // Create instances from the ARFF data
            ArffLoader.ArffReader arffReader = new ArffLoader.ArffReader(reader);
            data = arffReader.getData();
            
            // Set class index
            data.setClassIndex(data.numAttributes() - 1);
            
            LOGGER.info("Dataset loaded successfully with " + data.numInstances() + " instances");
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error loading dataset", e);
            throw e;
        }
    }
    
    // Step 2: Preprocess the Dataset
    private void preprocessDataset() throws Exception {
        // Convert the class attribute to numerical values
        NumericToNominal filter = new NumericToNominal();
        filter.setAttributeIndices("last");
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);
        
        // Randomize the dataset
        Randomize randomize = new Randomize();
        randomize.setInputFormat(data);
        data = Filter.useFilter(data, randomize);
        
        // Split the dataset into training and testing datasets
        RemovePercentage split = new RemovePercentage();
        split.setInputFormat(data);
        split.setPercentage(70);
        trainData = Filter.useFilter(data, split);
        split.setInvertSelection(true);
        testData = Filter.useFilter(data, split);
        
        LOGGER.info("Dataset preprocessing completed: " + 
                    trainData.numInstances() + " training instances, " + 
                    testData.numInstances() + " testing instances");
    }
    
    // Step 3: Train a Machine Learning Model
    private void trainModel() throws Exception {
        // Train a J48 decision tree on the training dataset
        classifier = new J48();
        classifier.buildClassifier(trainData);
        LOGGER.info("Model training completed with parameters: " + classifier.getOptions().toString());
    }
    
    // Step 4: Evaluate the Model
    private void evaluateModel() throws Exception {
        // Evaluate the model on the testing dataset
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(classifier, testData);
        
        // Store performance metrics
        accuracy = eval.pctCorrect();
        
        // Log the performance metrics
        LOGGER.info("Model evaluation completed: " + eval.toSummaryString());
        LOGGER.info("Model accuracy: " + accuracy + "%");
    }
    
    // Thread-safe getter for the classifier
    public J48 getClassifier() {
        rwLock.readLock().lock();
        try {
            return classifier;
        } finally {
            rwLock.readLock().unlock();
        }
    }
    
    // Thread-safe getter for the training data structure
    public Instances getTrainingDataStructure() {
        rwLock.readLock().lock();
        try {
            return trainData;
        } finally {
            rwLock.readLock().unlock();
        }
    }
    
    // Thread-safe getter for the full dataset structure
    public Instances getDataStructure() {
        rwLock.readLock().lock();
        try {
            return data;
        } finally {
            rwLock.readLock().unlock();
        }
    }
    
    // Get the model's current accuracy
    public double getModelAccuracy() {
        rwLock.readLock().lock();
        try {
            return accuracy;
        } finally {
            rwLock.readLock().unlock();
        }
    }
    
    // Get the last training time
    public String getLastTrainingTime() {
        rwLock.readLock().lock();
        try {
            return lastTrainingTime;
        } finally {
            rwLock.readLock().unlock();
        }
    }
    
    // Manually trigger model retraining - could be exposed through admin API
    public void triggerModelRetraining() throws Exception {
        LOGGER.info("Manual model retraining triggered");
        refreshModel();
        LOGGER.info("Manual model retraining completed successfully. New accuracy: " + accuracy);
    }
}