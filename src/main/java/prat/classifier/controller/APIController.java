package prat.classifier.controller;

import prat.classifier.model.InputFeatureRequest;
import prat.classifier.model.OutputTargetResponse;
import prat.classifier.service.ModelPredictionService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/classify")
public class APIController {
    
    @Autowired
    private ModelPredictionService modelUsageService;
    
    @GetMapping
    public String home() {
        return "CICD Pipeline V6-ML Springboot Microservice";
    }
    
    @PostMapping("/irisRequest")
    public ResponseEntity<OutputTargetResponse> classify(@RequestBody InputFeatureRequest irisData) {
        try {
            // Process the input data using the service layer
        	OutputTargetResponse classification = modelUsageService.classify(irisData);
            
            // Return the classification result
            return ResponseEntity.ok(classification);
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().build();
        }
    }
}
