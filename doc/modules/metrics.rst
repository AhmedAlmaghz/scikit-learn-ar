.. _metrics:

مقاييس الزوجية، التقاربات والنوى

```python
from sklearn.metrics import pairwise_distances
```

========================================

تحتوي وحدة فرعية :mod:`sklearn.metrics.pairwise` على أدوات مساعدة لتقييم المسافات الزوجية أو تقارب مجموعات العينات.

تحتوي هذه الوحدة على كل من مقاييس المسافة والنوى. يتم إعطاء ملخص موجز عن الاثنين هنا.

مقاييس المسافة هي دالات ``d(a, b)`` بحيث ``d(a, b) < d(a, c)`` إذا كانت الكائنات ``a`` و ``b`` تعتبر "أكثر تشابهًا" من الكائنات ``a`` و ``c``. سيكون لكائنين متطابقين تمامًا مسافة صفر. أحد الأمثلة الأكثر شيوعًا هو مسافة إقليدس. لكي تكون "حقيقية" ، يجب أن تلتزم بالمقاييس الأربعة التالية::

    1. d(a, b) >= 0, لجميع a و b
    2. d(a, b) == 0, إذا وفقط إذا a = b, التحديد الإيجابي
    3. d(a, b) == d(b, a), التناظر
    4. d(a, c) <= d(a, b) + d(b, c), عدم المساواة في المثلث

النوى هي مقاييس للتشابه ، أي ``s(a, b) > s(a, c)`` إذا كانت الكائنات ``a`` و ``b`` تعتبر "أكثر تشابهًا" من الكائنات ``a`` و ``c``. يجب أن تكون النواة أيضًا شبه محددة إيجابيًا.

هناك عدد من الطرق للتحويل بين مقياس المسافة ومقياس التشابه ، مثل النواة. لنفترض أن ``D`` هي المسافة ، و ``S`` هي النواة:

1. ``S = np.exp(-D * gamma)``، حيث أحد الإرشادات لاختيار ``gamma`` هو ``1 / num_features``
2. ``S = 1. / (D / np.max(D))``


.. currentmodule:: sklearn.metrics

يمكن تقييم المسافات بين متجهات الصف لـ ``X`` ومتجهات الصف لـ ``Y`` باستخدام :func:`pairwise_distances`. إذا تم حذف ``Y`` ، يتم حساب المسافات الزوجية لمتجهات الصف لـ ``X``. وبالمثل ، يمكن استخدام :func:`pairwise.pairwise_kernels` لحساب النواة بين `X` و `Y` باستخدام وظائف النواة المختلفة. راجع مرجع API لمزيد من التفاصيل.

    >>> import numpy as np
    >>> from sklearn.metrics import pairwise_distances
    >>> from sklearn.metrics.pairwise import pairwise_kernels
    >>> X = np.array([[2, 3], [3, 5], [5, 8]])
    >>> Y = np.array([[1, 0], [2, 1]])
    >>> pairwise_distances(X, Y, metric='manhattan')
    array([[ 4.,  2.],
           [ 7.,  5.],
           [12., 10.]])
    >>> pairwise_distances(X, metric='manhattan')
    array([[0., 3., 8.],
           [3., 0., 5.],
           [8., 5., 0.]])
    >>> pairwise_kernels(X, Y, metric='linear')
    array([[ 2.,  7.],
           [ 3., 11.],
           [ 5., 18.]])


.. currentmodule:: sklearn.metrics.pairwise

.. _cosine_similarity:

تشابه جيب التمام
-----------------
:func:`cosine_similarity` تحسب حاصل الضرب النقطي L2-normalized للمتجهات.
هذا يعني ، إذا كان :math:`x` و :math:`y` متجهات صف ،
يتم تعريف تشابه جيب التمام :math:`k` على النحو التالي:

.. math::

    k(x, y) = \frac{x y^\top}{\|x\| \|y\|}

يُطلق على هذا التشابه الجيبي ، لأن التطبيع الإقليدي (L2)
يعرض المتجهات على الكرة الوحدة ،
ثم يكون حاصل الضرب النقطي الخاص بهم هو جيب التمام للزاوية بين النقاط
التي تدل عليها المتجهات.

هذه النواة هي خيار شائع لحساب تشابه المستندات
الممثلة كمتجهات tf-idf.
:func:`cosine_similarity` تقبل مصفوفات ``scipy.sparse``.
(لاحظ أن وظيفة tf-idf في ``sklearn.feature_extraction.text``
يمكن أن تنتج متجهات طبيعية ، وفي هذه الحالة :func:`cosine_similarity`
تعادل :func:`linear_kernel` ، ولكن أبطأ.)

.. rubric:: المراجع

* C.D. Manning, P. Raghavan و H. Schütze (2008). مقدمة في
  استرجاع المعلومات. مطبعة جامعة كامبريدج.
 https://nlp.stanford.edu/IR-book/html/htmledition/the-vector-space-model-for-scoring-1.html

.. _linear_kernel:

النواة الخطية
-------------
تحسب الدالة :func:`linear_kernel` النواة الخطية ، وهي حالة خاصة من :func:`polynomial_kernel` مع ``degree=1`` و ``coef0=0`` (متجانسة).
إذا كان ``x`` و ``y`` متجهات عمود ، فإن النواة الخطية الخاصة بهم هي:

.. math::

    k(x, y) = x^\top y

.. _polynomial_kernel:

النواة متعددة الحدود
--------------------
تحسب الدالة :func:`polynomial_kernel` النواة متعددة الحدود من الدرجة d بين متجهين. تمثل النواة متعددة الحدود التشابه بين متجهين. من الناحية المفاهيمية ، تعتبر النواة متعددة الحدود ليس فقط التشابه بين المتجهات تحت نفس البعد ، ولكن أيضًا عبر الأبعاد. عند استخدامها في خوارزميات التعلم الآلي ، يسمح هذا بالحساب لتفاعل الميزة.

يتم تعريف النواة متعددة الحدود على النحو التالي:

.. math::

    k(x, y) = (\gamma x^\top y +c_0)^d

حيث:

* ``x``, ``y`` هي متجهات الإدخال
* ``d`` هي درجة النواة

إذا :math:`c_0 = 0` ، يقال أن النواة متجانسة.

.. _sigmoid_kernel:

نواة سيجمويد
------------
تحسب الدالة :func:`sigmoid_kernel` نواة سيجمويد بين متجهين. تُعرف نواة سيجمويد أيضًا باسم الظل الزائدي ، أو متعدد الطبقات (لأنه ، في مجال الشبكة العصبية ، غالبًا ما يتم استخدامه كدالة تنشيط الخلايا العصبية). يتم تعريفه على النحو التالي:

.. math::

    k(x, y) = \tanh( \gamma x^\top y + c_0)

حيث:

* ``x``, ``y`` هي متجهات الإدخال
* :math:`\gamma` معروفة باسم المنحدر
* :math:`c_0` معروفة باسم الاعتراض

.. _rbf_kernel:

نواة RBF
    

الوظيفة :func:`rbf_kernel` تحسب نواة الدالة الأساس الشعاعية (RBF) بين متجهين. يتم تعريف هذه النواة على النحو التالي:

.. math::

    k(x, y) = \exp( -\gamma \| x-y \|^2)

حيث ``x`` و ``y`` هما متجها الإدخال. إذا :math:`\gamma = \sigma^{-2}` فإن النواة تعرف باسم نواة غاوسية للتباين :math:`\sigma^2`.

.. _laplacian_kernel:

نواة لابلاسية
--------------

الوظيفة :func:`laplacian_kernel` هي نوع من نواة الدالة الأساس الشعاعية المعرفة على النحو التالي:

.. math::

    k(x, y) = \exp( -\gamma \| x-y \|_1)

حيث ``x`` و ``y`` هما متجها الإدخال و :math:`\|x-y\|_1` هي مسافة مانهاتن بين متجها الإدخال.

لقد ثبتت فائدتها في التعلم الآلي المطبق على البيانات الخالية من الضوضاء.
انظر على سبيل المثال `Machine learning for quantum mechanics in a nutshell
<https://onlinelibrary.wiley.com/doi/10.1002/qua.24954/abstract/>`_.

.. _chi2_kernel:

نواة كاي تربيع
--------------

نواة كاي تربيع هي خيار شائع جدًا لتدريب SVMs غير الخطية في تطبيقات الرؤية الحاسوبية.
يمكن حسابها باستخدام :func:`chi2_kernel` ثم تمريرها إلى :class:`~sklearn.svm.SVC` مع ``kernel="precomputed"``::

    >>> from sklearn.svm import SVC
    >>> from sklearn.metrics.pairwise import chi2_kernel
    >>> X = [[0, 1], [1, 0], [.2, .8], [.7, .3]]
    >>> y = [0, 1, 0, 1]
    >>> K = chi2_kernel(X, gamma=.5)
    >>> K
    array([[1.        , 0.36787944, 0.89483932, 0.58364548],
           [0.36787944, 1.        , 0.51341712, 0.83822343],
           [0.89483932, 0.51341712, 1.        , 0.7768366 ],
           [0.58364548, 0.83822343, 0.7768366 , 1.        ]])

    >>> svm = SVC(kernel='precomputed').fit(K, y)
    >>> svm.predict(K)
    array([0, 1, 0, 1])

يمكن أيضًا استخدامها مباشرة كوسيطة ``kernel``::

    >>> svm = SVC(kernel=chi2_kernel).fit(X, y)
    >>> svm.predict(X)
    array([0, 1, 0, 1])


نواة كاي تربيع تعطى بواسطة

.. math::

        k(x, y) = \exp \left (-\gamma \sum_i \frac{(x[i] - y[i]) ^ 2}{x[i] + y[i]} \right )

يفترض أن البيانات غير سالبة، وغالبًا ما يتم تطبيعها لتكون لها قيمة L1-norm تساوي واحدًا.
يتم تبرير التطبيع بالارتباط بمسافة كاي تربيع، وهي مسافة بين توزيعات الاحتمالات المنفصلة.

تستخدم نواة كاي تربيع بشكل شائع على التوزيعات (الأكياس) للكلمات المرئية.

.. rubric:: المراجع

* Zhang, J. and Marszalek, M. and Lazebnik, S. and Schmid, C.
  Local features and kernels for classification of texture and object
 categories: A comprehensive study
  International Journal of Computer Vision 2007
 https://hal.archives-ouvertes.fr/hal-00171412/document
    
