import cupy as cp
try:
    
    x = cp.random.rand(5)
   
    y = cp.dot(x, x)
    print(f"✅ RTX 4070 is fully connected! Result: {y}")
except Exception as e:
    print(f"❌ Still missing a library: {e}")
