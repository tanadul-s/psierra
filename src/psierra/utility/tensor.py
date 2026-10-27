from __future__ import annotations                                                                                                                                               
import numpy as np                                                                                                                                                               
                                                                                                                                                                                    
# Canonical Voigt index mapping for 3D symmetric tensors: (xx, yy, zz, xy, yz, zx)                                                                                               
VOIGT_MAP_3D: dict[int, tuple[int, int]] = {                                                                                                                                     
    0: (0, 0),  # xx                                                                                                                                                             
    1: (1, 1),  # yy                                                                                                                                                             
    2: (2, 2),  # zz                                                                                                                                                             
    3: (0, 1),  # xy                                                                                                                                                             
    4: (1, 2),  # yz                                                                                                                                                             
    5: (2, 0),  # zx                                                                                                                                                             
}                                                                                                                                                                                
                                                                                                                                                                                    
                                                                                                                                                                                    
def tensor_to_voigt_stress(tensor: np.ndarray) -> np.ndarray:                                                                                                                    
    """                                                                                                                                                                          
    Converts a symmetric 3x3 stress tensor to a 6D Voigt vector.                                                                                                                 
    Order: [sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx]                                                                                                          
                                                                                                                                                                                    
    Parameters:                                                                                                                                                                  
    -----------                                                                                                                                                                  
    tensor : np.ndarray of shape (3, 3)                                                                                                                                          
                                                                                                                                                                                    
    Returns:                                                                                                                                                                     
    --------                                                                                                                                                                     
    voigt : np.ndarray of shape (6,)                                                                                                                                             
    """                                                                                                                                                                          
    return np.array(                                                                                                                                                             
        [                                                                                                                                                                        
            tensor[0, 0],                                                                                                                                                        
            tensor[1, 1],                                                                                                                                                        
            tensor[2, 2],                                                                                                                                                        
            tensor[0, 1],                                                                                                                                                        
            tensor[1, 2],                                                                                                                                                        
            tensor[2, 0],                                                                                                                                                        
        ],                                                                                                                                                                       
        dtype=np.float64,                                                                                                                                                        
    )                                                                                                                                                                            
                                                                                                                                                                                    
                                                                                                                                                                                    
def tensor_to_voigt_strain(tensor: np.ndarray) -> np.ndarray:                                                                                                                    
    """                                                                                                                                                                          
    Converts a symmetric 3x3 strain tensor to a 6D Voigt engineering strain vector.                                                                                              
    Order: [eps_xx, eps_yy, eps_zz, 2*eps_xy, 2*eps_yz, 2*eps_zx]                                                                                                                
                                                                                                                                                                                    
    Parameters:                                                                                                                                                                  
    -----------                                                                                                                                                                  
    tensor : np.ndarray of shape (3, 3)                                                                                                                                          
                                                                                                                                                                                    
    Returns:                                                                                                                                                                     
    --------                                                                                                                                                                     
    voigt : np.ndarray of shape (6,)                                                                                                                                             
    """                                                                                                                                                                          
    return np.array(                                                                                                                                                             
        [                                                                                                                                                                        
            tensor[0, 0],                                                                                                                                                        
            tensor[1, 1],                                                                                                                                                        
            tensor[2, 2],                                                                                                                                                        
            2.0 * tensor[0, 1],                                                                                                                                                  
            2.0 * tensor[1, 2],                                                                                                                                                  
            2.0 * tensor[2, 0],                                                                                                                                                  
        ],                                                                                                                                                                       
        dtype=np.float64,                                                                                                                                                        
    )                                                                                                                                                                            
                                                                                                                                                                                    
                                                                                                                                                                                    
def voigt_stress_to_tensor(voigt: np.ndarray) -> np.ndarray:                                                                                                                     
    """                                                                                                                                                                          
    Converts a 6D Voigt stress vector to a symmetric 3x3 tensor.                                                                                                                 
                                                                                                                                                                                    
    Parameters:                                                                                                                                                                  
    -----------                                                                                                                                                                  
    voigt : np.ndarray of shape (6,)                                                                                                                                             
                                                                                                                                                                                    
    Returns:                                                                                                                                                                     
    --------                                                                                                                                                                     
    tensor : np.ndarray of shape (3, 3)                                                                                                                                          
    """
    tensor = np.zeros((3, 3), dtype=np.float64)
    tensor[0, 0] = voigt[0]
    tensor[1, 1] = voigt[1]
    tensor[2, 2] = voigt[2]
    tensor[0, 1] = tensor[1, 0] = voigt[3]
    tensor[1, 2] = tensor[2, 1] = voigt[4]
    tensor[2, 0] = tensor[0, 2] = voigt[5]
    return tensor
