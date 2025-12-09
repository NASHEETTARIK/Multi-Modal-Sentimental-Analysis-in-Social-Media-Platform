import pandas as pd
import numpy as np
import Performance 
import Performance_FS 
def Main_test():

    
    X_test_text = np.load('Features/X_test_text_enterface database.npy')
    
    X_test_audio = np.load('Features/X_test_audio_enterface database.npy')
    
    X_test_video1 =np.load('Features/X_test_video1_enterface database.npy')
    
    X_test_video2 = np.load('Features/X_test_video2_enterface database.npy')
    
    y_test = np.load('Features/y_test_enterface database.npy')    
    
    
    
    """========================================================================
                  Self Attention based Capsule Bi-lstm  
       ========================================================================"""   
    from SA_CBiLSTM_Text_model import Test_Text   
    pred_Text = Test_Text(X_test_text)    
    
    
    """========================================================================
               Existing Models - Bidirectional GRU ,GRU ,CNN  
       ========================================================================"""  
       
    from Existing_text import Existing_Test_Text1  
    Existing_pred_Text1 = Existing_Test_Text1(X_test_text)   
    
    
    from Existing_text import Existing_Test_Text2  
    Existing_pred_Text2 = Existing_Test_Text2(X_test_text)   
       
    
    from Existing_text import Existing_Test_Text3  
    Existing_pred_Text3 = Existing_Test_Text3(X_test_text)   
     
    
    """========================================================================
           Gated attention enclosed Residual context aware transformer  
       ========================================================================"""
    
    
    from GRCAT_Video_model import Test_Video   
    pred_Video = Test_Video(X_test_video1, X_test_video2)
    
    
    """========================================================================
             Existing Models - CNN-BiLSTM ,BiLSTM ,LDCNN      
       ========================================================================"""  
    
    from Existing_video import Existing_Test_Video1   
    Existing_pred_Video1 = Existing_Test_Video1(X_test_video1, X_test_video2)
    
    from Existing_video import Existing_Test_Video2   
    Existing_pred_Video2 = Existing_Test_Video2(X_test_video1, X_test_video2)
    
    
    from Existing_video import Existing_Test_Video3   
    Existing_pred_Video3 = Existing_Test_Video3(X_test_video1, X_test_video2)
    
    
    """========================================================================
             Densely connected recurrent network with dual Attention    
       ========================================================================"""  
       
    from DCRNA_Audio_model import Test_Audio   
    model_Audio = Test_Audio( X_test_audio)   
       
    """========================================================================
             Existing Models - resnet50,VGG16 ,LSTM     
       ========================================================================"""  
    
    from Existing_audio import Existing_Test_Audio1   
    Existing_pred_Audio1 = Existing_Test_Audio1( X_test_audio)  
    
    from Existing_audio import Existing_Test_Audio2  
    Existing_pred_Audio2 = Existing_Test_Audio2( X_test_audio)  
    
    from Existing_audio import Existing_Test_Audio3   
    Existing_pred_Audio3 = Existing_Test_Audio3(X_test_audio)  
    
    Performance.plot()


def Main_test_FS():
    
    X_test_text = np.load('Features1/X_test_text_enterface database.npy')
    
    X_test_audio = np.load('Features1/X_test_audio_enterface database.npy')
    
    X_test_video1 =np.load('Features1/X_test_video1_enterface database.npy')
    
    X_test_video2 = np.load('Features1/X_test_video2_enterface database.npy')
    
    y_test = np.load('Features1/y_test_enterface database.npy')    
    
    
    
    """========================================================================
                  Self Attention based Capsule Bi-lstm  
       ========================================================================"""   
    from SA_CBiLSTM_Text_model import Test_Text1   
    pred_Text = Test_Text1(X_test_text)    
    
    
    """========================================================================
               Existing Models - Bidirectional GRU ,GRU ,CNN  
       ========================================================================"""  
       
    from Existing_text import Existing_Test_Text11  
    Existing_pred_Text1 = Existing_Test_Text11(X_test_text)   
    
    
    from Existing_text import Existing_Test_Text22  
    Existing_pred_Text2 = Existing_Test_Text22(X_test_text)   
       
    
    from Existing_text import Existing_Test_Text33  
    Existing_pred_Text3 = Existing_Test_Text33(X_test_text)   
     
    
    """========================================================================
           Gated attention enclosed Residual context aware transformer  
       ========================================================================"""
    
    
    from GRCAT_Video_model import Test_Video1  
    pred_Video = Test_Video1(X_test_video1, X_test_video2)
    
    
    """========================================================================
             Existing Models - CNN-BiLSTM ,BiLSTM ,LDCNN      
       ========================================================================"""  
    
    from Existing_video import Existing_Test_Video11   
    Existing_pred_Video1 = Existing_Test_Video11(X_test_video1, X_test_video2)
    
    from Existing_video import Existing_Test_Video22   
    Existing_pred_Video2 = Existing_Test_Video22(X_test_video1, X_test_video2)
    
    
    from Existing_video import Existing_Test_Video33   
    Existing_pred_Video3 = Existing_Test_Video33(X_test_video1, X_test_video2)
    
    
    """========================================================================
             Densely connected recurrent network with dual Attention    
       ========================================================================"""  
       
    from DCRNA_Audio_model import Test_Audio1   
    model_Audio = Test_Audio1( X_test_audio)   
       
    """========================================================================
             Existing Models - resnet50,VGG16 ,LSTM     
       ========================================================================"""  
    
    from Existing_audio import Existing_Test_Audio11   
    Existing_pred_Audio1 = Existing_Test_Audio11( X_test_audio)  
    
    from Existing_audio import Existing_Test_Audio22  
    Existing_pred_Audio2 = Existing_Test_Audio22( X_test_audio)  
    
    from Existing_audio import Existing_Test_Audio33   
    Existing_pred_Audio3 = Existing_Test_Audio33(X_test_audio)  
    
    Performance_FS.plot()
    
if __name__ == "__main__":
    t = 1
    if t==0:
        Main_test()
    else:
        Main_test_FS()    
    
    
    
    
    
#%%%



# overAll Performance for Proposed: Video
# *******************

# Accuracy      :  0.9909
# Recall        :  0.973
# Precision     :  0.9734
# F Measure     :  0.9732
# Specificity   :  0.9945
# Sensitivity   :  0.973
# MAE           : 0.0545
# MSE           : 0.1401
# RMSE          : 0.3743



# overAll Performance for Proposed: Audio
# *******************

# Accuracy      :  0.9935
# Recall        :  0.9814
# Precision     :  0.9804
# F Measure     :  0.9809
# Specificity   :  0.9961
# Sensitivity   :  0.9814
# MAE           : 0.0506
# MSE           : 0.1673
# RMSE          : 0.409

# o

# overAll Performance for Proposed: Text
# *******************

# Accuracy      :  0.9844
# Recall        :  0.9532
# Precision     :  0.953
# F Measure     :  0.9531
# Specificity   :  0.9907
# Sensitivity   :  0.9532
# MAE           : 0.1012
# MSE           : 0.2879
# RMSE          : 0.5366


# overAll Performance for Proposed Fusion: 
# *******************

# Accuracy      :  0.9961
# Recall        :  0.9884
# Precision     :  0.9886
# F Measure     :  0.9885
# Specificity   :  0.9977
# Sensitivity   :  0.9884
# MAE           : 0.0272
# MSE           : 0.0817
# RMSE          : 0.2859

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

