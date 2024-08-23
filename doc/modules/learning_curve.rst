=====================================================
منحنيات التحقق: رسم الدرجات لتقييم النماذج
=====================================================

.. currentmodule:: sklearn.model_selection

كل مقدر له مزاياه وعيوبه. يمكن تحليل خطأ التعميم الخاص به من حيث الانحياز والتباين والضوضاء. **الانحياز** للمقدر هو متوسط الخطأ الخاص به لمجموعات تدريب مختلفة. **التباين** للمقدر يشير إلى مدى حساسيته لمجموعات تدريب مختلفة. الضوضاء هي خاصية للبيانات.

في الرسم البياني التالي، نرى دالة :math:`f(x) = \cos (\frac{3}{2} \pi x)` وبعض العينات الصاخبة من تلك الدالة. نستخدم ثلاثة مقادير مختلفة لتناسب الدالة: الانحدار الخطي مع الميزات متعددة الحدود من الدرجة 1، 4 و 15. نرى أن المقدار الأول يمكنه على أفضل تقدير توفير فقط تناسب ضعيف للعينات والدالة الحقيقية لأنه بسيط جدًا (انحياز عالي)، المقدار الثاني يقترب منه تقريبًا بشكل مثالي والمقدار الأخير يقترب من بيانات التدريب بشكل مثالي ولكنه لا يناسب الدالة الحقيقية بشكل جيد، أي أنه حساس جدًا لبيانات التدريب المتغيرة (تباين عالي).

.. figure:: ../auto_examples/model_selection/images/sphx_glr_plot_underfitting_overfitting_001.png
   :target: ../auto_examples/model_selection/plot_underfitting_overfitting.html
   :align: center
   :scale: 50%

الانحياز والتباين هما خاصيتان متأصلتان في المقدرين وعادةً ما يتعين علينا اختيار خوارزميات التعلم والمعلمات الفائقة بحيث يكون كل من الانحياز والتباين منخفضين قدر الإمكان (انظر `معضلة الانحياز-التباين <https://en.wikipedia.org/wiki/Bias-variance_dilemma>`_). طريقة أخرى لتقليل تباين النموذج هي استخدام المزيد من بيانات التدريب. ومع ذلك، يجب عليك فقط جمع المزيد من بيانات التدريب إذا كانت الدالة الحقيقية معقدة للغاية بحيث لا يمكن تقريبها بواسطة مقدر ذي تباين أقل.

في المشكلة الأحادية البسيطة التي رأيناها في المثال، من السهل معرفة ما إذا كان المقدر يعاني من الانحياز أو التباين. ومع ذلك، في المساحات عالية الأبعاد، يمكن أن تصبح النماذج صعبة للغاية للتصور. لهذا السبب، غالبًا ما يكون من المفيد استخدام الأدوات الموضحة أدناه.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_model_selection_plot_underfitting_overfitting.py`
* :ref:`sphx_glr_auto_examples_model_selection_plot_validation_curve.py`
* :ref:`sphx_glr_auto_examples_model_selection_plot_learning_curve.py`


.. _validation_curve:

منحنى التحقق
    
```
{
  "user_id": "user_id_value",
  "timestamp": "timestamp_value",
  "request": {
    "text": "هذا نص بتنسيق RST اريد منك أن تترجمه إلى اللغة العربية. اتبع التعليمات التالية في الترجمة :    لا تترجم كلا من : الرموز الخاصة, الرموز والمعادلات الرياضية, الروابط, التاجات, الشفرة البرمجية.    نص RST: \n\n.. _learning_curves:\n\n=====================================================\nValidation curves: plotting scores to evaluate models\n=====================================================\n\n.. currentmodule:: sklearn.model_selection\n\nEvery estimator has its advantages and drawbacks. Its generalization error\ncan be decomposed in terms of bias, variance and noise. The **bias** of an\nestimator is its average error for different training sets. The **variance**\nof an estimator indicates how sensitive it is to varying training sets. Noise\nis a property of the data.\n\nIn the following plot, we see a function :math:`f(x) = \\cos (\\frac{3}{2} \\pi x)`\nand some noisy samples from that function. We use three different estimators\nto fit the function: linear regression with polynomial features of degree 1,\n4 and 15. We see that the first estimator can at best provide only a poor fit\nto the samples and the true function because it is too simple (high bias),\nthe second estimator approximates it almost perfectly and the last estimator\napproximates the training data perfectly but does not fit the true function\nvery well, i.e. it is very sensitive to varying training data (high variance).\n\n.. figure:: ../auto_examples/model_selection/images/sphx_glr_plot_underfitting_overfitting_001.png\n   :target: ../auto_examples/model_selection/plot_underfitting_overfitting.html\n   :align: center\n   :scale: 50%\n\nBias and variance are inherent properties of estimators and we usually have to\nselect learning algorithms and hyperparameters so that both bias and variance\nare as low as possible (see `Bias-variance dilemma\n<https://en.wikipedia.org/wiki/Bias-variance_dilemma>`_). Another way to reduce\nthe variance of a model is to use more training data. However, you should only\ncollect more training data if the true function is too complex to be\napproximated by an estimator with a lower variance.\n\nIn the simple one-dimensional problem that we have seen in the example it is\neasy to see whether the estimator suffers from bias or variance. However, in\nhigh-dimensional spaces, models can become very difficult to visualize. For\nthis reason, it is often helpful to use the tools described below.\n\n.. rubric:: Examples\n\n* :ref:`sphx_glr_auto_examples_model_selection_plot_underfitting_overfitting.py`\n* :ref:`sphx_glr_auto_examples_model_selection_plot_validation_curve.py`\n* :ref:`sphx_glr_auto_examples_model_selection_plot_learning_curve.py`\n\n\n.. _validation_curve:\n\nValidation curve"
  },
  "response": {
    "text": "=====================================================\nمنحنيات التحقق: رسم الدرجات لتقييم النماذج\n=====================================================\n\n.. currentmodule:: sklearn.model_selection\n\nكل مقدر له مزاياه وعيوبه. يمكن تحليل خطأ التعميم الخاص به من حيث الانحياز والتباين والضوضاء. **الانحياز** للمقدر هو متوسط الخطأ الخاص به لمجموعات تدريب مختلفة. **التباين** للمقدر يشير إلى مدى حساسيته لمجموعات تدريب مختلفة. الضوضاء هي خاصية للبيانات.\n\nفي الرسم البياني التالي، نرى دالة :math:`f(x) = \\cos (\\frac{3}{2} \\pi x)` وبعض العينات الصاخبة من تلك الدالة. نستخدم ثلاثة مقادير مختلفة لتناسب الدالة: الانحدار الخطي مع الميزات متعددة الحدود من الدرجة 1، 4 و 15. نرى أن المقدار الأول يمكنه على أفضل تقدير توفير فقط تناسب ضعيف للعينات والدالة الحقيقية لأنه بسيط جدًا (انحياز عالي)، المقدار الثاني يقترب منه تقريبًا بشكل مثالي والمقدار الأخير يقترب من بيانات التدريب بشكل مثالي ولكنه لا يناسب الدالة الحقيقية بشكل جيد، أي أنه حساس جدًا لبيانات التدريب المتغيرة (تباين عالي).\n\n.. figure:: ../auto_examples/model_selection/images/sphx_glr_plot_underfitting_overfitting_001.png\n   :target: ../auto_examples/model_selection/plot_underfitting_overfitting.html\n   :align: center\n   :scale: 50%\n\nالانحياز والتباين هما خاصيتان متأصلتان في المقدرين وعادةً ما يتعين علينا اختيار خوارزميات التعلم والمعلمات الفائقة بحيث يكون كل من الانحياز والتباين منخفضين قدر الإمكان (انظر `معضلة الانحياز-التباين <https://en.wikipedia.org/wiki/Bias-variance_dilemma>`_). طريقة أخرى لتقليل تباين النموذج هي استخدام المزيد من بيانات التدريب. ومع ذلك، يجب عليك فقط جمع المزيد من بيانات التدريب إذا كانت الدالة الحقيقية معقدة للغاية بحيث لا يمكن تقريبها بواسطة مقدر ذي تباين أقل.\n\nفي المشكلة الأحادية البسيطة التي رأيناها في المثال، من السهل معرفة ما إذا كان المقدر يعاني من الانحياز أو التباين.
    
لتقييم نموذج، نحتاج إلى دالة تسجيل (انظر :ref:`model_evaluation`)، على سبيل المثال الدقة بالنسبة للمصنّفات. الطريقة الصحيحة لاختيار عدة معلمات فائقة لمقدر هي بالطبع البحث الشبكي أو طرق مشابهة (انظر :ref:`grid_search`) التي تختار المعلمة الفائقة ذات الدرجة القصوى على مجموعة التحقق أو عدة مجموعات التحقق. لاحظ أنه إذا قمنا بتحسين المعلمات الفائقة بناءً على درجة التحقق، فإن درجة التحقق تكون متحيزة وليست تقديرًا جيدًا للتعميم بعد الآن. للحصول على تقدير مناسب للتعميم، علينا حساب الدرجة على مجموعة اختبار أخرى.

ومع ذلك، من المفيد أحيانًا رسم تأثير معلمة فائقة واحدة على درجة التدريب ودرجة التحقق لمعرفة ما إذا كان المقدّر يعاني من الإفراط في التخصيص أو التخصيص الناقص لبعض قيم المعلمة الفائقة.

يمكن أن تساعد الدالة :func:`validation_curve` في هذه الحالة::

  >>> import numpy as np
  >>> from sklearn.model_selection import validation_curve
  >>> from sklearn.datasets import load_iris
  >>> from sklearn.svm import SVC

  >>> np.random.seed(0)
  >>> X, y = load_iris(return_X_y=True)
  >>> indices = np.arange(y.shape[0])
  >>> np.random.shuffle(indices)
  >>> X, y = X[indices], y[indices]

  >>> train_scores, valid_scores = validation_curve(
  ...     SVC(kernel="linear"), X, y, param_name="C", param_range=np.logspace(-7, 3, 3),
  ... )
  >>> train_scores
 array([[0.90..., 0.94..., 0.91..., 0.89..., 0.92...],
         [0.9... , 0.92..., 0.93..., 0.92..., 0.93...],
         [0.97..., 1...   , 0.98..., 0.97..., 0.99...]])
  >>> valid_scores
 array([[0.9..., 0.9... , 0.9... , 0.96..., 0.9... ],
         [0.9..., 0.83..., 0.96..., 0.96..., 0.93...],
         [1.... , 0.93..., 1....  , 1....  , 0.9... ]])

إذا كنت تنوي رسم منحنيات التحقق فقط، فإن الفئة :class:`~sklearn.model_selection.ValidationCurveDisplay` أكثر مباشرة من استخدام matplotlib يدويًا على نتائج استدعاء :func:`validation_curve`. يمكنك استخدام الطريقة :meth:`~sklearn.model_selection.ValidationCurveDisplay.from_estimator` على غرار :func:`validation_curve` لإنشاء ورسم منحنى التحقق:

.. plot::
   :context: close-figs
   :align: center

      from sklearn.datasets import load_iris
      from sklearn.model_selection import ValidationCurveDisplay
      from sklearn.svm import SVC
      from sklearn.utils import shuffle
      X, y = load_iris(return_X_y=True)
      X, y = shuffle(X, y, random_state=0)
      ValidationCurveDisplay.from_estimator(
         SVC(kernel="linear"), X, y, param_name="C", param_range=np.logspace(-7, 3, 10)
      )

إذا كانت درجة التدريب ودرجة التحقق منخفضتين، فإن المقدّر سيكون ناقص التخصيص. إذا كانت درجة التدريب عالية ودرجة التحقق منخفضة، فإن المقدّر يعاني من الإفراط في التخصيص، وإلا فإنه يعمل بشكل جيد جدًا. عادةً ما تكون درجة التدريب المنخفضة ودرجة التحقق العالية غير ممكنة. يتم عرض التخصيص الناقص، والإفراط في التخصيص، ونموذج العمل في الرسم البياني أدناه حيث نقوم بتغيير المعلمة `gamma` لمصنف SVM مع نواة RBF على مجموعة بيانات الأرقام.

.. figure:: ../auto_examples/model_selection/images/sphx_glr_plot_validation_curve_001.png
   :target: ../auto_examples/model_selection/plot_validation_curve.html
   :align: center
   :scale: 50%

.. _learning_curve:

منحنى التعلم
    

(ملاحظة: لم يتم تضمين الرسم البياني في الترجمة لأنه يتطلب تشغيل الكود.)

منحنى التعلم يعرض درجات التحقق والتدريب لمقدر ما لعدد مختلف من عينات التدريب. إنه أداة لمعرفة مدى استفادتنا من إضافة المزيد من بيانات التدريب وما إذا كان المقدر يعاني أكثر من خطأ التباين أو خطأ الانحياز. ضع في اعتبارك المثال التالي حيث نرسم منحنى التعلم لمصنف بايز الساذج وآلة المتجهات الداعمة (SVM).

بالنسبة لمصنف بايز الساذج، تتقارب كل من درجة التحقق ودرجة التدريب إلى قيمة منخفضة جدًا مع زيادة حجم مجموعة التدريب. وبالتالي، من المحتمل ألا نستفيد كثيرًا من المزيد من بيانات التدريب.

على النقيض من ذلك، بالنسبة لكميات صغيرة من البيانات، تكون درجة تدريب SVM أكبر بكثير من درجة التحقق. من المرجح أن يؤدي إضافة المزيد من عينات التدريب إلى زيادة التعميم.

.. figure:: ../auto_examples/model_selection/images/sphx_glr_plot_learning_curve_001.png
   :target: ../auto_examples/model_selection/plot_learning_curve.html
   :align: center
   :scale: 50%

يمكننا استخدام الدالة :func:`learning_curve` لتوليد القيم المطلوبة لرسم منحنى التعلم (عدد العينات التي تم استخدامها، ومتوسط الدرجات على مجموعات التدريب ومتوسط الدرجات على مجموعات التحقق)::

  >>> from sklearn.model_selection import learning_curve
  >>> from sklearn.svm import SVC

  >>> train_sizes, train_scores, valid_scores = learning_curve(
  ...     SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv=5)
  >>> train_sizes
 array([ 50, 80, 110])
  >>> train_scores
 array([[0.98..., 0.98 , 0.98..., 0.98..., 0.98...],
         [0.98..., 1.   , 0.98..., 0.98..., 0.98...],
         [0.98..., 1.   , 0.98..., 0.98..., 0.99...]])
  >>> valid_scores
 array([[1. ,  0.93...,  1. ,  1. ,  0.96...],
         [1. ,  0.96...,  1. ,  1. ,  0.96...],
         [1. ,  0.96...,  1. ,  1. ,  0.96...]])

إذا كنت تنوي رسم منحنيات التعلم فقط، فسيكون استخدام الفئة :class:`~sklearn.model_selection.LearningCurveDisplay` أسهل. يمكنك استخدام الطريقة :meth:`~sklearn.model_selection.LearningCurveDisplay.from_estimator` على غرار :func:`learning_curve` لتوليد ورسم منحنى التعلم:

.. plot::
   :context: close-figs
   :align: center

      from sklearn.datasets import load_iris
      from sklearn.model_selection import LearningCurveDisplay
      from sklearn.svm import SVC
      from sklearn.utils import shuffle
      X, y = load_iris(return_X_y=True)
      X, y = shuffle(X, y, random_state=0)
      LearningCurveDisplay.from_estimator(
         SVC(kernel="linear"), X, y, train_sizes=[50, 80, 110], cv=5)
    
