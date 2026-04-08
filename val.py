import warnings
warnings.filterwarnings('ignore')
from edgemaf import EdgeMAF

if __name__ == '__main__':

    model = EdgeMAF('runs/EdgeMAF-Net/test/weights/best.pt')
    metrics = model.val(
        data='E:/EdgeMAF-Netv2/dataset',
        imgsz=224,
        batch=64,
        workers=0,
        split='val',  
        plots=True,   
        save_json=True,  
        project='runs/EdgeMAF-Net',
        name='val_test',
    )

    print("\n" + "="*50)
    print("           验证结果")
    print("="*50)
    print(f"准确率 (Accuracy):      {metrics.accuracy:.4f}")
    print(f"精确率 (Precision):     {metrics.precision:.4f}")
    print(f"召回率 (Recall):        {metrics.recall:.4f}")
    print(f"F1分数 (F1-Score):      {metrics.f1_score:.4f}")
    print(f"特异性 (Specificity):   {metrics.specificity:.4f}")
    print(f"综合性能 (Fitness):     {metrics.fitness:.4f}")
    print("="*50)
