package prat.classifier.service;


import prat.classifier.Utility.ModelBuildTrainUtility;
import prat.classifier.model.InputFeatureRequest;
import prat.classifier.model.OutputTargetResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import weka.core.DenseInstance;


@Service
public class ModelPredictionService {
    
    @Autowired
    private ModelBuildTrainUtility modelBuildTrain;
    
    public OutputTargetResponse classify(InputFeatureRequest irisData) throws Exception {
        // Step 5: Make Predictions
        // Create a new instance to classify
        DenseInstance instance = new DenseInstance(4);
        instance.setValue(0, irisData.getSepalLength());
        instance.setValue(1, irisData.getSepalWidth());
        instance.setValue(2, irisData.getPetalLength());
        instance.setValue(3, irisData.getPetalWidth());
        
        // Set the reference dataset
        instance.setDataset(modelBuildTrain.getTrainingDataStructure());
        
        // Classify the new instance
        double classIndex = modelBuildTrain.getClassifier().classifyInstance(instance);
        
        // Get the class name
        String className = modelBuildTrain.getTrainingDataStructure().classAttribute().value((int) classIndex);
        
        // Get confidence (probability distribution)
        double[] distribution = modelBuildTrain.getClassifier().distributionForInstance(instance);
        double confidence = distribution[(int)classIndex] * 100;
        
        // Return the classification result
        System.out.println("Predicted class: " + className + " with confidence="+ confidence + "%");
        return new OutputTargetResponse(className, confidence);
    }
}