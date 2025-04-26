package prat.classifier;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.BeforeEach;

import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.mockito.junit.jupiter.MockitoExtension;

import prat.classifier.Utility.ModelBuildTrainUtility;
import prat.classifier.model.InputFeatureRequest;
import prat.classifier.model.OutputTargetResponse;
import prat.classifier.service.ModelPredictionService;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)  // do testing without starting app server
public class ModelPredictionServiceTest {

    @Mock
    private ModelBuildTrainUtility modelBuildTrainUtility;

    @Mock
    private J48 classifier;

    @InjectMocks
    private ModelPredictionService modelPredictionService;

    private Instances trainingDataStructure;

    //setup() constructs the necessary Weka data structures and configures the mock responses.
    @BeforeEach
    public void setUp() throws Exception { //
        MockitoAnnotations.openMocks(this);
        
        // Create attributes for the Iris dataset
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("sepallength"));
        attributes.add(new Attribute("sepalwidth"));
        attributes.add(new Attribute("petallength"));
        attributes.add(new Attribute("petalwidth"));
        
        // Create nominal class attribute with three values
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("Iris-setosa");
        classValues.add("Iris-versicolor");
        classValues.add("Iris-virginica");
        Attribute classAttribute = new Attribute("class", classValues);
        attributes.add(classAttribute);
        
        // Create instances object with the attributes
        trainingDataStructure = new Instances("iris", attributes, 0);
        trainingDataStructure.setClassIndex(trainingDataStructure.numAttributes() - 1);
        
        // Configure mock behavior
        when(modelBuildTrainUtility.getTrainingDataStructure()).thenReturn(trainingDataStructure);
        when(modelBuildTrainUtility.getClassifier()).thenReturn(classifier);
    }

    //Fast Execution: Tests run much faster without starting a web server or Spring context
   // Reliability: No network or external dependencies are required
    //Focus: Tests concentrate solely on ML logic without HTTP or other concerns
    //Isolation: Each component can be tested independently
    //Deterministic: Mock responses ensure consistent, predictable test outcomes
    @Test
    public void testClassifySetosa() throws Exception {
        // Setup mock classifier to return Setosa class (index 0)
        when(classifier.classifyInstance(any(DenseInstance.class))).thenReturn(0.0);
        
        // Setup mock probability distribution (90% confidence for Setosa)
        double[] distribution = {0.9, 0.05, 0.05};
        when(classifier.distributionForInstance(any(DenseInstance.class))).thenReturn(distribution);
        
        // Create sample input for Setosa
        InputFeatureRequest setosaInput = new InputFeatureRequest(5.1, 3.5, 1.4, 0.2);
        
        // Perform classification
        OutputTargetResponse result = modelPredictionService.classify(setosaInput);
        
        // Verify results
        assertEquals("Iris-setosa", result.getSpecies());
        assertEquals(90.0, result.getConfidence(), 0.01);
    }
}