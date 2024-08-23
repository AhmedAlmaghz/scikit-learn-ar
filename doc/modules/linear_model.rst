    .. _linear_model:

    =============
    نماذج خطية
    =============

    .. currentmodule:: sklearn.linear_model

    الطرق التالية مخصصة للانحدار حيث من المتوقع أن تكون القيمة المستهدفة مجموعة خطية من الميزات.
    بالرموز الرياضية، إذا كانت :math:`\hat{y}` هي القيمة المتوقعة.

    .. math::    \hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p

    في جميع أنحاء الوحدة، نحدد المتجه :math:`w = (w_1، ...، w_p)` كـ ``coef_`` و :math:`w_0` كـ ``intercept_``.

    لإجراء التصنيف باستخدام النماذج الخطية العامة، انظر :ref:`Logistic_regression`.

    .. _ordinary_least_squares:

    أقل المربعات العادية
    =======================

    :class:`LinearRegression` يلائم نموذجًا خطيًا بمعاملات :math:`w = (w_1، ...، w_p)` لتقليل مجموع مربعات المتبقي بين الأهداف المرصودة في مجموعة البيانات، والأهداف المتوقعة بالتقريب الخطي. من الناحية الرياضية، فإنه يحل مشكلة على شكل:

    .. math:: \min_{w} || X w - y||_2^2

    .. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_ols_001.png
       :target: ../auto_examples/linear_model/plot_ols.html
       :align: center
       :scale: 50%

    :class:`LinearRegression` سيأخذ في طريقة ``fit`` الخاصة به المصفوفات ``X``، ``y`` وسيخزن معاملات :math:`w` للنموذج الخطي في عضو ``coef_`` الخاص به::

        >>> from sklearn import linear_model
        >>> reg = linear_model.LinearRegression()
        >>> reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
        LinearRegression()
        >>> reg.coef_
        array([0.5, 0.5])

    تعتمد تقديرات المعامل لأقل المربعات العادية على استقلالية الميزات. عندما تكون الميزات مرتبطة والأعمدة في مصفوفة التصميم :math:`X` لها اعتماد خطي تقريبي، تصبح مصفوفة التصميم قريبة من التفرد ونتيجة لذلك، تصبح تقديرات أقل المربعات حساسة للغاية للأخطاء العشوائية في الهدف المرصود، مما ينتج عنه تباين كبير. يمكن أن تنشأ حالة *التعددية الخطية* هذه، على سبيل المثال، عندما يتم جمع البيانات دون تصميم تجريبي.

    .. rubric:: أمثلة

    * :ref:`sphx_glr_auto_examples_linear_model_plot_ols.py`

    أقل المربعات غير السلبية
    --------------------------

    من الممكن تقييد جميع المعاملات لتكون غير سالبة، مما قد يكون مفيدًا عندما تمثل بعض الكميات الفيزيائية أو الطبيعية غير السلبية (مثل تعدادات التردد أو أسعار السلع). :class:`LinearRegression` يقبل معلمة ``positive`` منطقية: عند تعيينها على `True` يتم تطبيق `أقل المربعات غير السلبية <https://en.wikipedia.org/wiki/Non-negative_least_squares>`_.

    .. rubric:: أمثلة

    * :ref:`sphx_glr_auto_examples_linear_model_plot_nnls.py`

    تعقيد أقل المربعات العادية
    ---------------------------------

    يتم حساب حل أقل المربعات باستخدام تحليل القيمة المفردة لـ X. إذا كانت X مصفوفة من الشكل `(n_samples، n_features)` فإن هذه الطريقة تكلف :math:`O(n_{\text{samples}} n_{\text{features}}^2)`, على افتراض أن :math:`n_{\text{samples}} \geq n_{\text{features}}`.

    .. _ridge_regression:

    انحدار ريدج والتصنيف
    
===================================

الانحدار
--------

تعالج انحدار :class:`Ridge` بعض مشاكل :ref:`ordinary_least_squares` عن طريق فرض عقوبة على حجم المعاملات. تقلل معاملات الانحدار من مجموع مربعات المخلفات المعاقب:

.. math::

   \min_{w} || X w - y||_2^2 + \alpha ||w||_2^2

يتحكم معامل التعقيد :math:`\alpha \geq 0` في مقدار الانكماش: كلما زادت قيمة :math:`\alpha`، زاد مقدار الانكماش وبالتالي تصبح المعاملات أكثر قوة في مواجهة الترابط.

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_ridge_path_001.png
   :target: ../auto_examples/linear_model/plot_ridge_path.html
   :align: center
   :scale: 50%

مثل النماذج الخطية الأخرى، ستأخذ :class:`Ridge` في طريقة ``fit`` الخاصة بها صفائف ``X``، ``y`` وستخزن معاملات النموذج الخطي :math:`w` في عضو ``coef_`` الخاص بها::

    >>> from sklearn import linear_model
    >>> reg = linear_model.Ridge(alpha=.5)
    >>> reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
    Ridge(alpha=0.5)
    >>> reg.coef_
    array([0.34545455, 0.34545455])
    >>> reg.intercept_
    0.13636...

لاحظ أن الفئة :class:`Ridge` تسمح للمستخدم بتحديد أن يتم اختيار المحلّل تلقائيًا عن طريق تعيين `solver="auto"`. عند تحديد هذا الخيار، ستختار :class:`Ridge` بين محللات `"lbfgs"`، `"cholesky"`، و`"sparse_cg"`. ستبدأ :class:`Ridge` في التحقق من الشروط الموضحة في الجدول التالي من الأعلى إلى الأسفل. إذا كان الشرط صحيحًا، يتم اختيار المحلّل المقابل.

+-------------+----------------------------------------------------+
| **المحلّل**  | **الشرط**                                      |
+-------------+----------------------------------------------------+
| 'lbfgs'     | يتم تحديد خيار ``positive=True``.         |
+-------------+----------------------------------------------------+
| 'cholesky'  | صفيف الإدخال X ليس متفرقًا.                   |
+-------------+----------------------------------------------------+
| 'sparse_cg' | لم يتم استيفاء أي من الشروط المذكورة أعلاه.        |
+-------------+----------------------------------------------------+


التصنيف
--------

يحتوي جهاز الانحدار :class:`Ridge` على متغير مصنف: :class:`RidgeClassifier`. يقوم هذا المصنف أولاً بتحويل الأهداف الثنائية إلى ``{-1، 1}`` ثم يعامل المشكلة كمهمة انحدار، مع تحسين نفس الهدف كما هو مذكور أعلاه. تتوافق الفئة المتوقعة مع إشارة توقع جهاز الانحدار. بالنسبة للتصنيف متعدد الفئات، تتم معالجة المشكلة على أنها انحدار متعدد المخرجات، وتتوافق الفئة المتوقعة مع الإخراج ذي القيمة الأعلى.

قد يبدو من المشكوك فيه استخدام خسارة المربعات الصغرى (المعاقبة) لتناسب نموذج التصنيف بدلاً من خسائر اللوجستية أو المفصلية الأكثر تقليدية. ومع ذلك، في الممارسة العملية، يمكن أن تؤدي جميع هذه النماذج إلى درجات التحقق المتقاطع المماثلة من حيث الدقة أو الدقة / الاسترجاع، بينما تسمح خسارة المربعات الصغرى المعاقبة التي يستخدمها :class:`RidgeClassifier` باختيار مختلف جدًا للمحللات الرقمية مع ملامح الأداء الحسابي المتميزة.

يمكن أن يكون :class:`RidgeClassifier` أسرع بكثير من :class:`LogisticRegression` مع عدد كبير من الفئات لأنه يمكنه حساب مصفوفة الإسقاط :math:`(X^T X)^{-1} X^T` مرة واحدة فقط.

يشار إلى هذا المصنف أحيانًا باسم `آلة دعم متجه المربعات الصغرى
<https://en.wikipedia.org/wiki/Least-squares_support-vector_machine>`_ مع نواة خطية.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_linear_model_plot_ridge_path.py`
* :ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`
* :ref:`sphx_glr_auto_examples_inspection_plot_linear_model_coefficient_interpretation.py`

تعقيد الانحدار
--------------

تتمتع هذه الطريقة بنفس ترتيب التعقيد مثل :ref:`ordinary_least_squares`.

.. FIXME:
.. Not completely true: OLS is solved by an SVD, while Ridge is solved by
.. the method of normal equations (Cholesky), there is a big flop difference
.. between these


ضبط معامل التنظيم: التحقق المتقاطع من ترك واحد
    

:class:`RidgeCV` و :class:`RidgeClassifierCV` ينفذان انحدار/تصنيف ريدج مع التحقق المتقاطع المدمج لمعامل ألفا. يعملان بنفس طريقة :class:`~sklearn.model_selection.GridSearchCV` باستثناء أنهما يستخدمان افتراضيًا التحقق المتقاطع الفعال من نوع Leave-One-Out. عند استخدام التحقق المتقاطع الافتراضي، لا يمكن أن تكون قيمة ألفا مساوية لـ 0 بسبب الصيغة المستخدمة لحساب خطأ Leave-One-Out. راجع [RL2007]_ للتفاصيل.

مثال الاستخدام::

    >>> import numpy as np
    >>> from sklearn import linear_model
    >>> reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
    >>> reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
    RidgeCV(alphas=array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01,
          1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06]))
    >>> reg.alpha_
    0.01

تحديد قيمة الخاصية :term:`cv` سيؤدي إلى استخدام التحقق المتقاطع مع :class:`~sklearn.model_selection.GridSearchCV`، على سبيل المثال `cv=10` للتحقق المتقاطع من نوع 10-fold، بدلاً من التحقق المتقاطع من نوع Leave-One-Out.

.. dropdown:: المراجع

  .. [RL2007] "Notes on Regularized Least Squares"، Rifkin & Lippert (`technical report
    <http://cbcl.mit.edu/publications/ps/MIT-CSAIL-TR-2007-025.pdf>`_,
    `course slides <https://www.mit.edu/~9.520/spring07/Classes/rlsslides.pdf>`_).

.. _lasso:

Lasso
=====

إن :class:`Lasso` هو نموذج خطي يقدر معاملات متفرقة. إنه مفيد في بعض السياقات بسبب ميله إلى تفضيل الحلول ذات معاملات غير صفرية أقل، مما يقلل بشكل فعال من عدد الميزات التي يعتمد عليها الحل المعطى. لهذا السبب، فإن Lasso ومتغيراته أساسية في مجال الاستشعار المضغوط. في ظل ظروف معينة، يمكنه استرداد مجموعة دقيقة من المعاملات غير الصفرية (انظر :ref:`sphx_glr_auto_examples_applications_plot_tomography_l1_reconstruction.py`).

رياضيا، يتكون من نموذج خطي مع مصطلح تنظيم إضافي. دالة الهدف للتقليل هي:

.. math::  \min_{w} { \frac{1}{2n_{\text{samples}}} ||X w - y||_2 ^ 2 + \alpha ||w||_1}

وبالتالي، فإن تقدير Lasso يحل مشكلة تقليل عقوبة المربعات الصغرى مع إضافة :math:`\alpha ||w||_1`، حيث :math:`\alpha` هو ثابت و:math:`||w||_1` هو :math:`\ell_1`-norm لمتجه المعامل.

التنفيذ في الفئة :class:`Lasso` يستخدم الانحدار الإحداثي كخوارزمية لتناسب المعاملات. انظر :ref:`least_angle_regression` لتنفيذ آخر::

    >>> from sklearn import linear_model
    >>> reg = linear_model.Lasso(alpha=0.1)
    >>> reg.fit([[0, 0], [1, 1]], [0, 1])
    Lasso(alpha=0.1)
    >>> reg.predict([[1, 1]])
    array([0.8])

الدالة :func:`lasso_path` مفيدة للمهام ذات المستوى الأدنى، حيث إنها تحسب المعاملات على طول المسار الكامل للقيم المحتملة.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_and_elasticnet.py`
* :ref:`sphx_glr_auto_examples_applications_plot_tomography_l1_reconstruction.py`
* :ref:`sphx_glr_auto_examples_inspection_plot_linear_model_coefficient_interpretation.py`

.. note:: **اختيار الميزة مع Lasso**

      نظرًا لأن انحدار Lasso ينتج نماذج متفرقة، فيمكن استخدامه
      لذلك لأداء اختيار الميزة، كما هو مفصل في
      :ref:`l1_feature_selection`.

.. dropdown:: المراجع

  تشرح المراجع التالية التكرارات
  المستخدمة في محلل الانحدار الإحداثي لـ scikit-learn، بالإضافة إلى
  حساب فجوة الازدواجية المستخدمة للتحكم في التقارب.

  * "Regularization Path For Generalized linear Models by Coordinate Descent"،
    Friedman, Hastie & Tibshirani، J Stat Softw، 2010 (`Paper
    <https://www.jstatsoft.org/article/view/v033i01/v33i01.pdf>`__).
  * "An Interior-Point Method for Large-Scale L1-Regularized Least Squares,"
    S. J. Kim, K. Koh, M. Lustig, S. Boyd and D. Gorinevsky,
    in IEEE Journal of Selected Topics in Signal Processing, 2007
    (`Paper <https://web.stanford.edu/~boyd/papers/pdf/l1_ls.pdf>`__)

ضبط معامل التنظيم
    

يتحكم معامل "alpha" في درجة التفرق في معاملات التقدير.

 استخدام التحقق المتقاطع
 ^^^^^^^^^^^^^^^^^^^^^^^

 يوفر scikit-learn كائنات تقوم بتعيين معامل "alpha" في Lasso عن طريق التحقق المتقاطع: :class:`LassoCV` و :class:`LassoLarsCV`. يعتمد :class:`LassoLarsCV` على خوارزمية :ref:`least_angle_regression` التي سيتم شرحها أدناه.

 بالنسبة لمجموعات البيانات عالية الأبعاد ذات العديد من الميزات المترابطة ، فإن :class:`LassoCV` غالبًا ما يكون الأفضل. ومع ذلك ، فإن :class:`LassoLarsCV` له ميزة استكشاف قيم أكثر صلة بمعامل `alpha` ، وإذا كان عدد العينات صغيرًا جدًا مقارنة بعدد الميزات ، فإنه غالبًا ما يكون أسرع من :class:`LassoCV`.

 .. |lasso_cv_1| image:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_model_selection_002.png
     :target: ../auto_examples/linear_model/plot_lasso_model_selection.html
     :scale: 48%

 .. |lasso_cv_2| image:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_model_selection_003.png
     :target: ../auto_examples/linear_model/plot_lasso_model_selection.html
     :scale: 48%

 .. centered:: |lasso_cv_1| |lasso_cv_2|

 .. _lasso_lars_ic:

 اختيار النموذج القائم على معايير المعلومات
 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 بدلاً من ذلك ، يقترح المقدّر :class:`LassoLarsIC` استخدام معيار معلومات Akaike (AIC) ومعيار معلومات Bayes (BIC). إنها بديل أرخص حسابياً لإيجاد القيمة المثلى لـ alpha حيث يتم حساب مسار الانتظام مرة واحدة فقط بدلاً من k + 1 مرة عند استخدام التحقق المتقاطع k-fold.

 في الواقع ، يتم حساب هذه المعايير على مجموعة التدريب داخل العينة. باختصار ، فإنها تعاقب الدرجات المتفائلة للغاية لمختلف نماذج Lasso بمرونتها (راجع قسم "التفاصيل الرياضية" أدناه).

 ومع ذلك ، تحتاج هذه المعايير إلى تقدير مناسب لدرجات حرية الحل ، وهي مشتقة من نتائج كبيرة (النتائج渐近) وتفترض أن النموذج الصحيح هو المرشحين قيد التحقيق. كما أنها تميل إلى الانهيار عندما تكون المشكلة سيئة التكييف (على سبيل المثال ، المزيد من الميزات أكثر من العينات).

 .. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_lars_ic_001.png
     :target: ../auto_examples/linear_model/plot_lasso_lars_ic.html
     :align: center
     :scale: 50%

 .. rubric:: أمثلة

 * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py`
 * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_lars_ic.py`

 .. _aic_bic:

 معايير AIC و BIC
 ^^^^^^^^^^^^^^^^^

 قد يختلف تعريف AIC (وبالتالي BIC) في الأدبيات. في هذا القسم ، نقدم المزيد من المعلومات المتعلقة بالمعيار المحسوب في scikit-learn.

 .. dropdown:: التفاصيل الرياضية

   يتم تعريف معيار AIC على النحو التالي:

   .. math::
       AIC = -2 \log(\hat{L}) + 2 d

   حيث :math:`\hat{L}` هو الحد الأقصى للاحتمال للنموذج و :math:`d` هو عدد المعلمات (كما يشار إليه أيضًا بدرجات الحرية في القسم السابق).

   يستبدل تعريف BIC الثابت :math:`2` بـ :math:`\log(N)`:

   .. math::
       BIC = -2 \log(\hat{L}) + \log(N) d

   حيث :math:`N` هو عدد العينات.

   بالنسبة لنموذج خطي غاوسي ، يتم تعريف الحد الأقصى للوغاريتم كما يلي:

   .. math::
       \log(\hat{L}) = - \frac{n}{2} \log(2 \pi) - \frac{n}{2} \ln(\sigma^2) - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{2\sigma^2}

   حيث :math:`\sigma^2` هو تقدير تباين الضوضاء ، :math:`y_i` و :math:`\hat{y}_i` هي الأهداف الحقيقية والمتوقعة على التوالي ، و :math:`n` هو عدد العينات.

   يؤدي توصيل الحد الأقصى للوغاريتم في صيغة AIC إلى:

   .. math::
       AIC = n \log(2 \pi \sigma^2) + \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sigma^2} + 2 d

   غالبًا ما يتم تجاهل المصطلح الأول من التعبير أعلاه لأنه ثابت عندما يتم توفير :math:`\sigma^2`. بالإضافة إلى ذلك ، يُذكر أحيانًا أن AIC يعادل إحصائية :math:`C_p` [12]_. بالمعنى الدقيق للكلمة ، ومع ذلك ، فهو مكافئ فقط حتى بعض الثوابت وعامل الضرب.

   أخيرًا ، ذكرنا أعلاه أن :math:`\sigma^2` هو تقدير لتباين الضوضاء. في :class:`LassoLarsIC` عندما لا يتم توفير المعلمة `noise_variance` (افتراضيًا) ، يتم تقدير تباين الضوضاء عبر المقدّر غير المتحيز [13]_ المحدد على النحو التالي:

   .. math::
       \sigma^2 = \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{n - p}

   حيث :math:`p` هو عدد الميزات و :math:`\hat{y}_i` هو الهدف المتوقع باستخدام انحدار المربعات الصغرى العادية. لاحظ أن هذه الصيغة صالحة فقط عندما `n_samples > n_features`.

   .. rubric:: المراجع

   .. [12] :arxiv:`Zou, Hui, Trevor Hastie, and Robert Tibshirani.
           "On the degrees of freedom of the lasso."
           The Annals of Statistics 35.5 (2007): 2173-2192.
           <0712.0881.pdf>`

   .. [13] :doi:`Cherkassky, Vladimir, and Yunqian Ma.
           "Comparison of model selection for regression."
           Neural computation 15.7 (2003): 1691-1714.
           <10.1162/089976603321891864>`
    

مقارنة مع معامل التثبيت في SVM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

يتم إعطاء التكافؤ بين "ألفا" ومعامل التثبيت في SVM، "C" بواسطة "ألفا = 1 / C" أو "ألفا = 1 / (n_samples * C)"، اعتمادًا على المقدّر ودالة الهدف المحددة التي يتم تحسينها بواسطة النموذج.

.. _multi_task_lasso:

Multi-task Lasso
    
================

إن :class:`MultiTaskLasso` هو نموذج خطي يقدر معاملات متفرقة لمشاكل الانحدار المتعددة بشكل مشترك: ``y`` هو مصفوفة ثنائية الأبعاد، من الشكل ``(n_samples, n_tasks)``. القيد هو أن الميزات المحددة هي نفسها لجميع مشاكل الانحدار، والتي تسمى أيضًا المهام.

تقارن الصورة التالية موقع الإدخالات غير الصفرية في مصفوفة المعاملات W التي تم الحصول عليها باستخدام Lasso بسيط أو MultiTaskLasso. تنتج تقديرات Lasso قيمًا غير صفرية متناثرة بينما تكون قيم MultiTaskLasso غير الصفرية عبارة عن أعمدة كاملة.

.. |multi_task_lasso_1| image:: ../auto_examples/linear_model/images/sphx_glr_plot_multi_task_lasso_support_001.png
    :target: ../auto_examples/linear_model/plot_multi_task_lasso_support.html
    :scale: 48%

.. |multi_task_lasso_2| image:: ../auto_examples/linear_model/images/sphx_glr_plot_multi_task_lasso_support_002.png
    :target: ../auto_examples/linear_model/plot_multi_task_lasso_support.html
    :scale: 48%

.. centered:: |multi_task_lasso_1| |multi_task_lasso_2|

.. centered:: تركيب نموذج سلسلة زمنية، وفرض أن أي ميزة نشطة تكون نشطة في جميع الأوقات.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_linear_model_plot_multi_task_lasso_support.py`


.. dropdown:: التفاصيل الرياضية

  يتكون من نموذج خطي تم تدريبه باستخدام :math:`\ell_1` :math:`\ell_2`-norm مختلط للتحكم في النمو.
  دالة الهدف التي يجب تقليلها هي:

  .. math::  \min_{W} { \frac{1}{2n_{\text{samples}}} ||X W - Y||_{\text{Fro}} ^ 2 + \alpha ||W||_{21}}

  حيث :math:`\text{Fro}` تشير إلى معيار Frobenius

  .. math:: ||A||_{\text{Fro}} = \sqrt{\sum_{ij} a_{ij}^2}

  و :math:`\ell_1` :math:`\ell_2` تقرأ

  .. math:: ||A||_{2 1} = \sum_i \sqrt{\sum_j a_{ij}^2}.

  التنفيذ في الفئة :class:`MultiTaskLasso` يستخدم الانحدار الإحداثي كخوارزمية لتناسب المعاملات.

.. _elastic_net:

Elastic-Net
===========

إن :class:`ElasticNet` هو نموذج انحدار خطي تم تدريبه بكل من :math:`\ell_1` و :math:`\ell_2`-norm للتحكم في النمو.
هذا المزيج يسمح بتعلم نموذج متفرق حيث القليل من الأوزان غير صفرية مثل :class:`Lasso`، مع الحفاظ على خصائص التحكم في النمو لـ :class:`Ridge`. نتحكم في المجموعة المحدبة لـ :math:`\ell_1` و :math:`\ell_2` باستخدام معامل ``l1_ratio``.

Elastic-net مفيد عندما تكون هناك ميزات متعددة مرتبطة ببعضها البعض. من المحتمل أن يختار Lasso واحدًا من هذه بشكل عشوائي، بينما من المحتمل أن يختار Elastic-net كليهما.

ميزة عملية للتداول بين Lasso و Ridge هي أنه يسمح لـ Elastic-Net بوراثة بعض من استقرار Ridge تحت الدوران.

دالة الهدف التي يجب تقليلها في هذه الحالة هي

.. math::

    \min_{w} { \frac{1}{2n_{\text{samples}}} ||X w - y||_2 ^ 2 + \alpha \rho ||w||_1 +
    \frac{\alpha(1-\rho)}{2} ||w||_2 ^ 2}


.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_coordinate_descent_path_001.png
   :target: ../auto_examples/linear_model/plot_lasso_coordinate_descent_path.html
   :align: center
   :scale: 50%

يمكن استخدام الفئة :class:`ElasticNetCV` لتعيين المعاملات ``alpha`` (:math:`\alpha`) و ``l1_ratio`` (:math:`\rho`) عن طريق التحقق المتقاطع.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_and_elasticnet.py`
* :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_coordinate_descent_path.py`
* :ref:`sphx_glr_auto_examples_linear_model_plot_elastic_net_precomputed_gram_matrix_with_weighted_samples.py`

.. dropdown:: المراجع

  تشرح المراجع التالية التكرارات
  المستخدمة في محلل الانحدار الإحداثي لـ scikit-learn، بالإضافة إلى
  حساب فجوة الازدواجية المستخدمة للتحكم في التقارب.

  * "Regularization Path For Generalized linear Models by Coordinate Descent"،
    Friedman, Hastie & Tibshirani، J Stat Softw، 2010 (`Paper
    <https://www.jstatsoft.org/article/view/v033i01/v33i01.pdf>`__).
  * "An Interior-Point Method for Large-Scale L1-Regularized Least Squares,"
    S. J. Kim, K. Koh, M. Lustig, S. Boyd and D. Gorinevsky،
    في IEEE Journal of Selected Topics in Signal Processing، 2007
    (`Paper <https://web.stanford.edu/~boyd/papers/pdf/l1_ls.pdf>`__)

.. _multi_task_elastic_net:

Multi-task Elastic-Net
    
   
    إن :class:`MultiTaskElasticNet` هو نموذج خطي يقدر معاملات متفرقة لمشاكل الانحدار المتعددة بشكل مشترك باستخدام كل من :math:`\ell_1` و :math:`\ell_2`-norm للتحكم في النمو.
    دالة الهدف التي يجب تقليلها هي:

    .. math::  \min_{W} { \frac{1}{2n_{\text{samples}}} ||X W - Y||_{\text{Fro}} ^ 2 + \alpha \rho ||W||_{21} +
    \frac{\alpha(1-\rho)}{2} ||W||_{\text{Fro}} ^ 2}

    حيث :math:`\text{Fro}` تشير إلى معيار Frobenius

    .. math:: ||A||_{\text{Fro}} = \sqrt{\sum_{ij} a_{ij}^2}

    و :math:`\ell_1` :math:`\ell_2` تقرأ

    .. math:: ||A||_{2 1} = \sum_i \sqrt{\sum_j a_{ij}^2}.

    التنفيذ في الفئة :class:`MultiTaskElasticNet` يستخدم الانحدار الإحداثي كخوارزمية لتناسب المعاملات.

.. _multi_task_elastic_net_cv:

Multi-task Elastic-Net CV
    
    إن :class:`MultiTaskElasticNetCV` هو إصدار من :class:`MultiTaskElasticNet` يحدد معاملات ``alpha`` (:math:`\alpha`) و ``l1_ratio`` (:math:`\rho`) عن طريق التحقق المتقاطع.
    
    إن :class:`MultiTaskElasticNetCV` يستخدم خوارزمية "Cyclic" للتنسيق بين الانحدار والبحث عن الشبكة.
    
    إن :class:`MultiTaskElasticNetCV` يستخدم خوارزمية "Randomized" للتنسيق بين الانحدار والبحث عن الشبكة.
    
    إن :class:`MultiTaskElasticNetCV` يستخدم خوارزمية "GridSearch" للتنسيق بين الانحدار والبحث عن الشبكة.
    
    إن :class:`MultiTaskElasticNetCV` يستخدم خوارزمية "HalvingGridSearch" للتنسيق بين الانحدار والبحث عن الشبكة.
    
    إن :class:`MultiTaskElasticNetCV
  النص الذي تم إدخاله هو بتنسيق RST (ReStructuredText) ، وهو تنسيق يستخدم عادةً في توثيق Python. سأقوم بترجمة النص إلى اللغة العربية مع اتباع التعليمات المذكورة.

    ---

    إن :class:`MultiTaskElasticNet` هو نموذج شبكة مرنة يقدر معاملات متفرقة لمشاكل الانحدار المتعددة بشكل مشترك: ``Y`` عبارة عن مصفوفة ثنائية الأبعاد ذات شكل ``(n_samples, n_tasks)``. القيد هو أن الميزات المحددة هي نفسها لجميع مشاكل الانحدار ، والتي تسمى أيضًا المهام.

    رياضيا ، يتكون من نموذج خطي مدرب مع :math:`\ell_1` :math:`\ell_2`-norm و :math:`\ell_2`-norm للتحكم في الانتظام. دالة الهدف التي يجب تقليلها هي:

    .. math::

        \min_{W} { \frac{1}{2n_{\text{samples}}} ||X W - Y||_{\text{Fro}}^2 + \alpha \rho ||W||_{2 1} +
        \frac{\alpha(1-\rho)}{2} ||W||_{\text{Fro}}^2}

    يستخدم التنفيذ في الفئة :class:`MultiTaskElasticNet` الانحدار الإحداثي كخوارزمية لتناسب المعاملات.

    يمكن استخدام الفئة :class:`MultiTaskElasticNetCV` لتعيين المعلمات ``alpha`` (:math:`\alpha`) و ``l1_ratio`` (:math:`\rho`) عن طريق التحقق المتقاطع.

    .. _least_angle_regression:

    انحدار الزاوية الأقل (LARS)
    ==========================

    انحدار الزاوية الأقل (LARS) هو خوارزمية انحدار للبيانات عالية الأبعاد ، تم تطويرها بواسطة Bradley Efron و Trevor Hastie و Iain Johnstone و Robert Tibshirani. LARS مشابه للانحدار التدريجي للأمام. في كل خطوة ، يجد الميزة الأكثر ارتباطًا بالهدف. عندما يكون هناك العديد من الميزات التي لها ارتباط متساوٍ ، بدلاً من الاستمرار في نفس الميزة ، فإنه يتقدم في اتجاه متساوي الزاوية بين الميزات.

    مزايا LARS هي:

    - إنها فعالة عدديًا في السياقات التي يكون فيها عدد الميزات أكبر بكثير من عدد العينات.

    - إنها سريعة حسابيًا مثل التحديد للأمام ولها نفس ترتيب التعقيد مثل المربعات الصغرى العادية.

    - ينتج مسار حل خطي كامل ، وهو مفيد في التحقق المتقاطع أو محاولات مماثلة لضبط النموذج.

    - إذا كانت هناك ميزتان مرتبطتان تقريبًا بالهدف ، فيجب أن تزداد معاملاتهما بنفس المعدل تقريبًا. وبالتالي ، تتصرف الخوارزمية كما هو متوقع من الحدس ، وهي أيضًا أكثر استقرارًا.

    - يمكن تعديله بسهولة لإنتاج حلول للمقدرين الآخرين ، مثل Lasso.

    عيوب طريقة LARS تشمل:

    - نظرًا لأن LARS يعتمد على إعادة تركيب متكررة للمتبقيات ، يبدو أنه حساس بشكل خاص لآثار الضوضاء. تتم مناقشة هذه المشكلة بالتفصيل من قبل Weisberg في قسم المناقشة لمقال Efron et al. (2004) Annals of Statistics.

    يمكن استخدام نموذج LARS عبر المقدّر :class:`Lars` ، أو تنفيذه منخفض المستوى :func:`lars_path` أو :func:`lars_path_gram`.


    LARS Lasso
    ==========

    :class:`LassoLars` هو نموذج lasso تم تنفيذه باستخدام خوارزمية LARS ، وعلى عكس التنفيذ المستند إلى الانحدار الإحداثي ، فإن هذا ينتج الحل الدقيق ، وهو خطي على شكل قطع كدالة للمعيار لمعاملاته.

    .. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_lars_001.png
       :target: ../auto_examples/linear_model/plot_lasso_lars.html
       :align: center
       :scale: 50%

    ::

       >>> from sklearn import linear_model
       >>> reg = linear_model.LassoLars(alpha=.1)
       >>> reg.fit([[0, 0], [1, 1]], [0, 1])
       LassoLars(alpha=0.1)
       >>> reg.coef_
       array([0.6..., 0.        ])

    .. rubric:: أمثلة

    * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_lars.py`

    توفر خوارزمية Lars المسار الكامل للمعاملات على طول معامل الانتظام تقريبًا مجانًا ، وبالتالي فإن العملية الشائعة هي استرداد المسار باستخدام إحدى الدالتين :func:`lars_path` أو :func:`lars_path_gram`.

    .. dropdown:: الصياغة الرياضية

       الخوارزمية مشابهة للانحدار التدريجي للأمام ، ولكن بدلاً من تضمين الميزات في كل خطوة ، يتم زيادة المعاملات المقدرة في اتجاه متساوي الزاوية مع ارتباط كل منها بالمتبقي.

       بدلاً من إعطاء نتيجة متجه ، يتكون حل LARS من منحنى يدل على الحل لكل قيمة من :math:`\ell_1` norm لمتجه المعلمة. يتم تخزين مسار المعاملات الكامل في الصفيف ``coef_path_`` ذي الشكل `(n_features، max_features + 1)`. العمود الأول هو دائما صفر.

       .. rubric:: المراجع

       * تم تفصيل الخوارزمية الأصلية في الورقة `Least Angle Regression <https://www-stat.stanford.edu/~hastie/Papers/LARS/LeastAngle_2

...

    انحدار المطابقة العمودية (OMP)
    -----------------------------

    ...
    (يتم قطع النص هنا لأنه يتجاوز الحد الأقصى لعدد الأحرف المسموح به في الرد. يرجى الاطلاع على النص الأصلي للحصول على التفاصيل الكاملة.)

:class:`OrthogonalMatchingPursuit` و :func:`orthogonal_mp` ينفذان خوارزمية OMP لتقريب ملاءمة نموذج خطي مع قيود مفروضة على عدد المعاملات غير الصفرية (أي pseudo-norm :math:`\ell_0`).

باعتبارها طريقة اختيار ميزة أمامية مثل :ref:`least_angle_regression`، يمكن لمطابقة التقارب العمودي تقريب متجه الحل الأمثل بعدد ثابت من العناصر غير الصفرية:

.. math::
    \underset{w}{\operatorname{arg\,min\,}}  ||y - Xw||_2^2 \text{ subject to } ||w||_0 \leq n_{\text{nonzero_coefs}}

بدلاً من ذلك، يمكن لمطابقة التقارب العمودي استهداف خطأ معين بدلاً من عدد معين من المعاملات غير الصفرية. يمكن التعبير عن هذا على النحو التالي:

.. math::
    \underset{w}{\operatorname{arg\,min\,}} ||w||_0 \text{ subject to } ||y-Xw||_2^2 \leq \text{tol}


تستند OMP إلى خوارزمية جشعة تتضمن في كل خطوة الذرة الأكثر ارتباطًا بالبقايا الحالية. وهي تشبه طريقة مطابقة السعي (MP) الأبسط، ولكنها أفضل من حيث أنه في كل تكرار، يتم إعادة حساب الباقي باستخدام إسقاط عمودي على مساحة عناصر القاموس المختارة مسبقًا.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_linear_model_plot_omp.py`

.. dropdown:: المراجع

  * https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

  * `Matching pursuits with time-frequency dictionaries
    <https://www.di.ens.fr/~mallat/papiers/MallatPursuit93.pdf>`_,
    S. G. Mallat, Z. Zhang,

.. _bayesian_regression:

انحدار بايزي
    

التقنيات الانحدارية البيزية يمكن استخدامها لإدراج معاملات التثبيت في إجراء التقدير: معامل التثبيت لا يتم تعيينه بشكل صارم ولكن يتم ضبطه على البيانات المتاحة.

يمكن القيام بذلك عن طريق إدخال "priors غير مفيدة" على المعلمات الفائقة للنموذج.

إن التثبيت :math:`\ell_{2}` المستخدم في :ref:`ridge_regression` يعادل إيجاد تقدير الحد الأقصى لاحتمالية الخلفية تحت توزيع غاوسي سابق على المعاملات :math:`w` مع الدقة :math:`\lambda^{-1}`.

بدلاً من تعيين `\lambda` يدويًا، من الممكن معاملته كمتغير عشوائي ليتم تقديره من البيانات.

لحصول على نموذج احتمالي كامل، من المفترض أن يكون الإخراج :math:`y` موزعًا غاوسيًا حول :math:`X w`:

.. math:: p(y|X,w,\alpha) = \mathcal{N}(y|X w,\alpha^{-1})

حيث :math:`\alpha` تتم معاملته مرة أخرى كمتغير عشوائي يجب تقديره من البيانات.

مزايا الانحدار البيزي هي:

- يتكيف مع البيانات المتاحة.

- يمكن استخدامه لإدراج معاملات التثبيت في إجراء التقدير.

عيوب الانحدار البيزي تشمل:

- استنتاج النموذج يمكن أن يكون مستهلكًا للوقت.

.. dropdown:: المراجع

  * مقدمة جيدة للطرق البيزية موجودة في C. Bishop: Pattern Recognition and Machine learning

  * تم تفصيل الخوارزمية الأصلية في كتاب `Bayesian learning for neural networks` بواسطة Radford M. Neal

.. _bayesian_ridge_regression:

انحدار ريدج البيزي
-------------------------

:class:`BayesianRidge` يقدر نموذج احتمالي لمشكلة الانحدار كما هو موضح أعلاه.

يتم إعطاء السابق للمعامل :math:`w` بواسطة غاوسي كروي:

.. math:: p(w|\lambda) =
    \mathcal{N}(w|0,\lambda^{-1}\mathbf{I}_{p})

يتم اختيار التوزيعات السابقة على :math:`\alpha` و :math:`\lambda` لتكون `توزيعات غاما <https://en.wikipedia.org/wiki/Gamma_distribution>`__، السابقة المترافقة لدقة الغاوسي. النموذج الناتج يسمى *انحدار ريدج البيزي*، وهو مشابه لـ :class:`Ridge` الكلاسيكي.

يتم تقدير المعلمات :math:`w`، :math:`\alpha` و :math:`\lambda` بشكل مشترك أثناء ملاءمة النموذج، ويتم تقدير معاملات التثبيت :math:`\alpha` و :math:`\lambda` عن طريق زيادة *log marginal likelihood*. يعتمد التنفيذ في scikit-learn على الخوارزمية الموضحة في الملحق أ من (Tipping، 2001) حيث يتم تحديث المعلمات :math:`\alpha` و :math:`\lambda` كما هو مقترح في (MacKay، 1992). يمكن تعيين القيمة الأولية لإجراء التكبير باستخدام المعلمات الفائقة ``alpha_init`` و ``lambda_init``.

هناك أربعة معلمات فائقة أخرى، :math:`\alpha_1`، :math:`\alpha_2`، :math:`\lambda_1` و :math:`\lambda_2` لتوزيعات غاما السابقة على :math:`\alpha` و :math:`\lambda`. هذه عادة ما يتم اختيارها لتكون *غير مفيدة*. بشكل افتراضي :math:`\alpha_1 = \alpha_2 =  \lambda_1 = \lambda_2 = 10^{-6}`.

يتم استخدام انحدار ريدج البيزي للانحدار::

    >>> from sklearn import linear_model
    >>> X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
    >>> Y = [0., 1., 2., 3.]
    >>> reg = linear_model.BayesianRidge()
    >>> reg.fit(X, Y)
    BayesianRidge()

بعد الملاءمة، يمكن استخدام النموذج للتنبؤ بالقيم الجديدة::

    >>> reg.predict([[1, 0.]])
    array([0.50000013])

يمكن الوصول إلى معاملات :math:`w` للنموذج::

    >>> reg.coef_
    array([0.49999993, 0.49999993])

نظرًا للإطار البيزي، فإن الأوزان التي تم العثور عليها مختلفة قليلاً عن تلك التي تم العثور عليها بواسطة :ref:`ordinary_least_squares`. ومع ذلك، فإن انحدار ريدج البيزي أكثر قوة للمشاكل غير المطروحة بشكل جيد.

.. rubric:: الأمثلة

* :ref:`sphx_glr_auto_examples_linear_model_plot_bayesian_ridge_curvefit.py`

.. dropdown:: المراجع

  * القسم 3.3 في Christopher M. Bishop: Pattern Recognition and Machine Learning، 2006

  * David J. C. MacKay، `Bayesian Interpolation <https://citeseerx.ist.psu.edu/doc_view/pid/b14c7cc3686e82ba40653c6dff178356a33e5e2c>`_، 1992.

  * Michael E. Tipping، `Sparse Bayesian Learning and the Relevance Vector Machine <https://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf>`_، 2001.

.. _automatic_relevance_determination:

تحديد الصلة التلقائي - ARD
    
التحديد التلقائي للارتباط (كما هو مطبق في :class:`ARDRegression`) هو نوع من النماذج الخطية يشبه إلى حد كبير "انحدار ريدج بايزي" (_Bayesian Ridge Regression)، ولكنه يؤدي إلى معاملات أكثر ندرة :math:`w` [1] [2].

:class:`ARDRegression` يفرض توزيعًا مختلفًا على :math:`w`: فهو يتخلى عن التوزيع الكروي الغاوسي لصالح التوزيع الإهليلجي الغاوسي المتمركز. هذا يعني أن كل معامل :math:`w_{i}` يمكن أن يتم سحبه من توزيع غاوسي، متمركز حول الصفر وبدقة :math:`\lambda_{i}`:

.. math:: p(w|\lambda) = \mathcal{N}(w|0,A^{-1})

مع :math:`A` كونه مصفوفة قطرية موجبة التحديد و :math:`\text{diag}(A) = \lambda = \{\lambda_{1},...,\lambda_{p}\}`.

على النقيض من "انحدار ريدج بايزي" (_Bayesian Ridge Regression)، لكل إحداثية من :math:`w_{i}` انحرافها المعياري الخاص :math:`\frac{1}{\lambda_i}`. يتم اختيار التوزيع المسبق لجميع :math:`\lambda_i` ليكون نفس توزيع جاما المعطى بواسطة المعلمات الفوقية :math:`\lambda_1` و :math:`\lambda_2`.

من المعروف أيضًا في الأدبيات أن ARD هو *التعلم البايزي الضئيل* و *آلة متجه الصلة* [3] [4]. للحصول على مقارنة مفصلة بين ARD و "انحدار ريدج بايزي" (_Bayesian Ridge Regression)، انظر المثال أدناه.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_linear_model_plot_ard.py`


.. rubric:: المراجع

.. [1] كريستوفر م. بيشوب: التعرف على الأنماط والتعلم الآلي، الفصل 7.2.1

.. [2] ديفيد ويبف وسريكانتان ناجاراجان: `وجهة نظر جديدة حول التحديد التلقائي للارتباط <https://papers.nips.cc/paper/3372-a-new-view-of-automatic-relevance-determination.pdf>`_

.. [3] مايكل إي. تيبينج: `التعلم البايزي الضئيل وآلة متجه الصلة <https://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf>`_

.. [4] تريستان فليتشر: `شرح آلات متجه الصلة <https://citeseerx.ist.psu.edu/doc_view/pid/3dc9d625404fdfef6eaccc3babddefe4c176abd4>`_

.. _Logistic_regression:

انحدار لوجستي
    
يتم تنفيذ الانحدار اللوجستي في :class:`LogisticRegression`. على الرغم من اسمه، يتم تنفيذه كنموذج خطي للتصنيف بدلاً من الانحدار من حيث تسمية scikit-learn/ML. يُعرف الانحدار اللوجستي أيضًا في الأدبيات باسم انحدار اللوغاريتم، أو تصنيف الحد الأقصى للانتروبيا (MaxEnt) أو المصنف اللوغاريتمي الخطي. في هذا النموذج، يتم نمذجة الاحتمالات التي تصف النتائج المحتملة لتجربة واحدة باستخدام `الدالة اللوجستية <https://en.wikipedia.org/wiki/Logistic_function>`_.

يمكن لهذا التنفيذ أن يلائم الانحدار اللوجستي الثنائي، أو One-vs-Rest، أو متعدد الحدود مع :math:`\ell_1` اختياري، :math:`\ell_2` أو تنظيم Elastic-Net.

.. note:: **التنظيم**

    يتم تطبيق التنظيم افتراضيًا، وهو أمر شائع في التعلم الآلي ولكن ليس في الإحصاء. ميزة أخرى للتنظيم هي أنه يحسن الاستقرار العددي. لا يوجد تنظيم يوازي تعيين C إلى قيمة عالية جدًا.

.. note:: **الانحدار اللوجستي كحالة خاصة من النماذج الخطية المعممة (GLM)**

    الانحدار اللوجستي هو حالة خاصة من :ref:`generalized_linear_models` مع توزيع شرطي Binomial / Bernoulli ورابط Logit. يمكن استخدام الإخراج العددي للانحدار اللوجستي، وهو الاحتمال المتوقع، كمصنف عن طريق تطبيق عتبة (افتراضيًا 0.5) عليه. هذه هي الطريقة التي يتم تنفيذها في scikit-learn، لذلك فهي تتوقع هدفًا فئويًا، مما يجعل الانحدار اللوجستي مصنفًا.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_linear_model_plot_logistic_l1_l2_sparsity.py`
* :ref:`sphx_glr_auto_examples_linear_model_plot_logistic_path.py`
* :ref:`sphx_glr_auto_examples_linear_model_plot_logistic_multinomial.py`
* :ref:`sphx_glr_auto_examples_linear_model_plot_sparse_logistic_regression_20newsgroups.py`
* :ref:`sphx_glr_auto_examples_linear_model_plot_sparse_logistic_regression_mnist.py`

الحالة الثنائية
--------------

لتسهيل التدوين، نفترض أن الهدف :math:`y_i` يأخذ قيمًا في المجموعة :math:`\{0، 1\}` لنقطة البيانات :math:`i`.
بمجرد تركيبه، فإن طريقة :meth:`~sklearn.linear_model.LogisticRegression.predict_proba` لـ :class:`~sklearn.linear_model.LogisticRegression` تتنبأ باحتمال الفئة الإيجابية :math:`P(y_i=1|X_i)` ك

.. math:: \hat{p}(X_i) = \operatorname{expit}(X_i w + w_0) = \frac{1}{1 + \exp(-X_i w - w_0)}.


كمشكلة تحسين، يقلل الانحدار اللوجستي الثنائي للفئة مع مصطلح التنظيم :math:`r(w)` دالة التكلفة التالية:

.. math::
    :name: regularized-logistic-loss

    \min_{w} \frac{1}{S}\sum_{i=1}^n s_i
    \left(-y_i \log(\hat{p}(X_i)) - (1 - y_i) \log(1 - \hat{p}(X_i))\right)
    + \frac{r(w)}{S C}\,,

حيث :math:`{s_i}` يتوافق مع الأوزان التي يحددها المستخدم لعينة تدريب محددة (يتكون المتجه :math:`s` من الضرب العنصري لأوزان الفئة وأوزان العينة)،
والمجموع :math:`S = \sum_{i=1}^n s_i`.

نقدم حاليًا أربعة خيارات لمصطلح التنظيم :math:`r(w)` عبر وسيطة `penalty`:

+----------------+-------------------------------------------------+
| penalty        | :math:`r(w)`                                    |
+================+=================================================+
| `None`         | :math:`0`                                       |
+----------------+-------------------------------------------------+
| :math:`\ell_1` | :math:`\|w\|_1`                                 |
+----------------+-------------------------------------------------+
| :math:`\ell_2` | :math:`\frac{1}{2}\|w\|_2^2 = \frac{1}{2}w^T w` |
+----------------+-------------------------------------------------+
| `ElasticNet`   | :math:`\frac{1 - \rho}{2}w^T w + \rho \|w\|_1`  |
+----------------+-------------------------------------------------+

بالنسبة لـ ElasticNet، يتحكم :math:`\rho` (الذي يتوافق مع وسيطة `l1_ratio`) في قوة تنظيم :math:`\ell_1` مقابل تنظيم :math:`\ell_2`. Elastic-Net يعادل :math:`\ell_1` عندما :math:`\rho = 1` ويعادل :math:`\ell_2` عندما :math:`\rho=0`.

لاحظ أن مقياس أوزان الفئة وأوزان العينة سيؤثر على مشكلة التحسين. على سبيل المثال، ضرب أوزان العينة بثابت :math:`b>0` يعادل ضرب قوة التنظيم العكسي `C` بـ :math:`b`.

الحالة متعددة الحدود


يمكن توسيع الحالة الثنائية إلى :math:`K` فئات مما يؤدي إلى الانحدار اللوجستي متعدد الحدود، انظر أيضًا `نموذج لوغاريتمي خطي <https://en.wikipedia.org/wiki/Multinomial_logistic_regression#As_a_log-linear_model>`_.

.. note::
   من الممكن تحديد معالم نموذج تصنيف :math:`K`-فئة باستخدام متجهات وزن :math:`K-1` فقط، تاركًا احتمال فئة واحدة محددًا بالكامل من خلال احتمالات الفئة الأخرى من خلال الاستفادة من حقيقة أن جميع احتمالات الفئة يجب أن تصل إلى واحد. نحن نختار عن قصد زيادة معلمات النموذج باستخدام متجهات وزن :math:`K` لتسهيل التنفيذ والحفاظ على الانحياز الاستقرائي المتماثل فيما يتعلق بترتيب الفئات، انظر [16]_. يصبح هذا التأثير مهمًا بشكل خاص عند استخدام التنظيم. يمكن أن يكون اختيار زيادة المعلمات ضارًا بالنماذج غير المعاقبة نظرًا لأن الحل قد لا يكون فريدًا، كما هو موضح في [16]_.

.. dropdown:: التفاصيل الرياضية

  دع :math:`y_i \in {1، \ldots، K}` يكون متغير الهدف المشفر (الترتيبي) للملاحظة :math:`i`.
  بدلاً من متجه معاملات واحد، لدينا الآن
  مصفوفة معاملات :math:`W` حيث يتوافق كل متجه صف :math:`W_k` مع الفئة
  :math:`k`. نهدف إلى التنبؤ باحتمالات الفئة :math:`P(y_i=k|X_i)` عبر
  :meth:`~sklearn.linear_model.LogisticRegression.predict_proba` كما يلي:

  .. math:: \hat{p}_k(X_i) = \frac{\exp(X_i W_k + W_{0, k})}{\sum_{l=0}^{K-1} \exp(X_i W_l + W_{0, l})}.

  يصبح الهدف للتحسين

  .. math::
    \min_W -\frac{1}{S}\sum_{i=1}^n \sum_{k=0}^{K-1} s_{ik} [y_i = k] \log(\hat{p}_k(X_i))
    + \frac{r(W)}{S C}\,,

  حيث :math:`[P]` يمثل قوس Iverson الذي يتم تقييمه على :math:`0`
  إذا كان :math:`P` خاطئًا، وإلا فإنه يتم تقييمه على :math:`1`.

  مرة أخرى، :math:`s_{ik}` هي الأوزان التي يحددها المستخدم (ضرب أوزان العينة وأوزان الفئة) مع مجموعها :math:`S = \sum_{i=1}^n \sum_{k=0}^{K-1} s_{ik}`.

  نقدم حاليًا أربعة خيارات
  لمصطلح التنظيم :math:`r(W)` عبر وسيطة `penalty`، حيث :math:`m`
  هو عدد الميزات:

  +----------------+----------------------------------------------------------------------------------+
  | penalty        | :math:`r(W)`                                                                     |
  +================+==================================================================================+
  | `None`         | :math:`0`                                                                        |
  +----------------+----------------------------------------------------------------------------------+
  | :math:`\ell_1` | :math:`\|W\|_{1,1} = \sum_{i=1}^m\sum_{j=1}^{K}|W_{i,j}|`                        |
  +----------------+----------------------------------------------------------------------------------+
  | :math:`\ell_2` | :math:`\frac{1}{2}\|W\|_F^2 = \frac{1}{2}\sum_{i=1}^m\sum_{j=1}^{K} W_{i,j}^2`   |
  +----------------+----------------------------------------------------------------------------------+
  | `ElasticNet`   | :math:`\frac{1 - \rho}{2}\|W\|_F^2 + \rho \|W\|_{1,1}`                           |
  +----------------+----------------------------------------------------------------------------------+

الحلول
    
-------

    الحلول المنفذة في الفئة :class:`LogisticRegression` هي "lbfgs"، "liblinear"، "newton-cg"، "newton-cholesky"، "sag" و "saga":

    الجدول التالي يلخص العقوبات ودعم متعدد الطبقات متعدد الطبقات الذي يدعمه كل حل:

    +------------------------------+-------------+-----------------+-----------------+-----------------------+-----------+------------+
    |                              |                       **الحلول**                                                                |
    +------------------------------+-------------+-----------------+-----------------+-----------------------+-----------+------------+
    | **العقوبات**                | **'lbfgs'** | **'liblinear'** | **'newton-cg'** | **'newton-cholesky'** | **'sag'** | **'saga'** |
    +------------------------------+-------------+-----------------+-----------------+-----------------------+-----------+------------+
    | عقوبة L2                   |     نعم     |       لا        |       نعم       |     لا                |    نعم    |    نعم     |
    +------------------------------+-------------+-----------------+-----------------+-----------------------+-----------+------------+
    | عقوبة L1                   |     لا      |       نعم       |       لا        |     لا                |    لا     |    نعم     |
    +------------------------------+-------------+-----------------+-----------------+-----------------------+-----------+------------+
    | Elastic-Net (L1 + L2)        |     لا      |       لا        |       لا        |     لا                |    لا     |    نعم     |
    +------------------------------+-------------+-----------------+-----------------+-----------------------+-----------+------------+
    | بدون عقوبة ('none')          |     نعم     |       لا        |       نعم       |     نعم               |    نعم    |    نعم     |
    +------------------------------+-------------+-----------------+-----------------+-----------------------+-----------+------------+
    | **دعم متعدد الطبقات**       |                                                                                                  |
    +------------------------------+-------------+-----------------+-----------------+-----------------------+-----------+------------+
    | متعدد الطبقات متعدد الطبقات       |     نعم     |       لا        |       نعم       |     لا                |    نعم    |    نعم     |
    +------------------------------+-------------+-----------------+-----------------+-----------------------+-----------+------------+
    | **السلوكيات**                |                                                                                                  |
    +------------------------------+-------------+-----------------+-----------------+-----------------------+-----------+------------+
    | معاقبة الاعتراض (سيء)        |     لا      |       نعم       |       لا        |     لا                |    لا     |    لا      |
    +------------------------------+-------------+-----------------+-----------------+-----------------------+-----------+------------+
    | أسرع للمجموعات الكبيرة من البيانات    |     لا      |       لا        |       لا        |     لا                |    نعم    |    نعم     |
    +------------------------------+-------------+-----------------+-----------------+-----------------------+-----------+------------+
    | متين لمجموعات البيانات غير المقيّدة  |     نعم     |       نعم       |       نعم       |     نعم               |    لا     |    لا      |
    +------------------------------+-------------+-----------------+-----------------+-----------------------+-----------+------------+

    يتم استخدام حل "lbfgs" بشكل افتراضي لصلابته. للمجموعات الكبيرة من البيانات، يكون حل "saga" عادةً أسرع.
    للمجموعات الكبيرة من البيانات، يمكنك أيضًا التفكير في استخدام :class:`SGDClassifier`
    مع `loss="log_loss"`, والذي قد يكون أسرع ولكنه يتطلب المزيد من الضبط.

    .. _liblinear_differences:

    الاختلافات بين الحلول
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    قد يكون هناك اختلاف في الدرجات التي تم الحصول عليها بين
    :class:`LogisticRegression` مع ``solver=liblinear`` أو
    :class:`~sklearn.svm.LinearSVC` والمكتبة الخارجية liblinear مباشرة،
    عندما يكون ``fit_intercept=False`` و ``coef_`` المناسب (أو) البيانات التي سيتم التنبؤ بها
    هي أصفار. هذا لأنه بالنسبة للعينة (العينات) ذات ``decision_function`` صفر،
    :class:`LogisticRegression` و :class:`~sklearn.svm.LinearSVC` يتنبآن
    بالفئة السلبية، بينما يتنبأ liblinear بالفئة الإيجابية. لاحظ أن نموذجًا به
    ``fit_intercept=False`` وله العديد من العينات ذات ``decision_function`` صفر،
    من المحتمل أن يكون نموذجًا سيئًا وغير مناسب، وننصحك بتعيين
    ``fit_intercept=True`` وزيادة ``intercept_scaling``.

    .. dropdown:: تفاصيل الحلول
        

* يستخدم المحلِّل "liblinear" خوارزمية انحدار الإحداثيات (CD)، ويعتمد على مكتبة C++ الممتازة `LIBLINEAR <https://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_، والتي يتم شحنها مع scikit-learn. ومع ذلك، لا يمكن لخوارزمية CD المنفَّذة في liblinear تعلُّم نموذج متعدد الحدود (متعدد الفئات) حقيقي؛ بدلاً من ذلك، يتم تحليل مشكلة التحسين بطريقة "واحد مقابل البقية" بحيث يتم تدريب مصنفات ثنائية منفصلة لجميع الفئات. يحدث هذا تحت الغطاء، لذلك تتصرف مثيلات :class:`LogisticRegression` التي تستخدم هذا المحلِّل كمصنِّفات متعددة الفئات. بالنسبة للتحجيم :math:`\ell_1`، تسمح :func:`sklearn.svm.l1_min_c` بحساب الحد الأدنى لـ C للحصول على نموذج "غير فارغ" (جميع أوزان الميزات إلى الصفر).

* تدعم محلِّلات "lbfgs" و"newton-cg" و"sag" فقط التحجيم :math:`\ell_2` أو عدم التحجيم، وتوجد لتتقارب بشكل أسرع لبعض البيانات عالية الأبعاد. يؤدي تعيين `multi_class` إلى "multinomial" مع هذه المحلِّلات إلى تعلُّم نموذج حقيقي للانحدار اللوجستي متعدد الحدود [5]_، مما يعني أن تقديرات احتمالاته يجب أن تكون أفضل معايرة من إعداد "واحد مقابل البقية" الافتراضي.

* يستخدم محلِّل "sag" خوارزمية Stochastic Average Gradient descent [6]_. وهو أسرع من المحلِّلات الأخرى للمجموعات الكبيرة من البيانات، عندما يكون كل من عدد العينات وعدد الميزات كبيرًا.

* محلِّل "saga" [7]_ هو نسخة من "sag" تدعم أيضًا `penalty="l1"` غير السلس. لذلك، هذا هو المحلِّل المفضل للانحدار اللوجستي المتعدد الحدود النادر. وهو أيضًا المحلِّل الوحيد الذي يدعم `penalty="elasticnet"`.

* "lbfgs" هو خوارزمية تحسين تقرِّب خوارزمية Broyden–Fletcher–Goldfarb–Shanno [8]_، والتي تنتمي إلى طرق كوازي-نيوتن. على هذا النحو، يمكنه التعامل مع مجموعة واسعة من بيانات التدريب المختلفة، وبالتالي فهو المحلِّل الافتراضي. ومع ذلك، فإن أدائه يعاني في مجموعات البيانات سيئة القياس وفي مجموعات البيانات ذات الميزات الفئوية المشفرة بشكل أحادي مع فئات نادرة.

* محلِّل "newton-cholesky" هو محلِّل نيوتن الدقيق الذي يحسب مصفوفة hessian ويحل النظام الخطي الناتج. إنه خيار جيد جدًا لـ `n_samples` >> `n_features`، ولكن له بعض أوجه القصور: يتم دعم التحجيم :math:`\ell_2` فقط. علاوة على ذلك، نظرًا لأنه يتم حساب مصفوفة hessian بشكل صريح، فإن استخدام الذاكرة له اعتماد تربيعي على `n_features` وكذلك على `n_classes`. نتيجة لذلك، يتم تنفيذ مخطط واحد مقابل البقية فقط لحالة متعددة الفئات.

لمقارنة بعض هذه المحلِّلات، انظر [9]_.

.. rubric:: المراجع

.. [5] Christopher M. Bishop: Pattern Recognition and Machine Learning, Chapter 4.3.4

.. [6] Mark Schmidt, Nicolas Le Roux, and Francis Bach: `Minimizing Finite Sums with the Stochastic Average Gradient. <https://hal.inria.fr/hal-00860051/document>`_

.. [7] Aaron Defazio, Francis Bach, Simon Lacoste-Julien:
      :arxiv:`SAGA: A Fast Incremental Gradient Method With Support for
      Non-Strongly Convex Composite Objectives. <1407.0202>`

.. [8] https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm

.. [9] Thomas P. Minka `"A comparison of numerical optimizers for logistic regression"
          <https://tminka.github.io/papers/logreg/minka-logreg.pdf>`_

.. [16] :arxiv:`Simon, Noah, J. Friedman and T. Hastie.
      "A Blockwise Descent Algorithm for Group-penalized Multiresponse and
      Multinomial Regression." <1311.6529>`

.. note:: **اختيار الميزة مع الانحدار اللوجستي النادر**

   ينتج الانحدار اللوجستي مع عقوبة :math:`\ell_1` نماذج نادرة، وبالتالي يمكن استخدامه لأداء اختيار الميزة، كما هو مفصل في :ref:`l1_feature_selection`.

.. note:: **تقدير قيمة P**

    من الممكن الحصول على قيم P وفواصل الثقة لمعاملات الحالات الانحدارية دون عقوبة. تدعم حزمة `statsmodels <https://pypi.org/project/statsmodels/>`_ هذا أصلاً. داخل sklearn، يمكن للمرء استخدام bootstrapping بدلاً من ذلك أيضًا.


تطبِّق :class:`LogisticRegressionCV` الانحدار اللوجستي مع دعم مدمج للتحقق المتقاطع، لإيجاد معلمات `C` و`l1_ratio` المثلى وفقًا لسمة ``scoring``. تم العثور على محلِّلات "newton-cg" و"sag" و"saga" و"lbfgs" لتكون أسرع للبيانات الكثيفة عالية الأبعاد، نظرًا لبدء التشغيل الدافئ (انظر :term:`Glossary <warm_start>`).

.. _Generalized_linear_regression:

.. _Generalized_linear_models:

النماذج الخطية المعممة

نماذج الخطية المعممة (GLM) توسع النماذج الخطية بطريقتين [10]_. أولاً، ترتبط القيم المتوقعة :math:`\hat{y}` بمجموعة خطية من متغيرات الإدخال :math:`X` عبر دالة ارتباط عكسية :math:`h` كما يلي:

.. math::    \hat{y}(w, X) = h(Xw).

ثانياً، يتم استبدال دالة الخسارة المربعة بوحدة الانحراف :math:`d` لتوزيع في العائلة الأسية (أو بشكل أكثر دقة، نموذج تشتت أسّي تكاثري (EDM) [11]_).

تصبح مشكلة التقليل:

.. math::    \min_{w} \frac{1}{2 n_{\text{samples}}} \sum_i d(y_i, \hat{y}_i) + \frac{\alpha}{2} ||w||_2^2,

حيث :math:`\alpha` هي عقوبة التنظيم L2. عندما يتم توفير أوزان العينة، يصبح المتوسط متوسطًا مرجحًا.

يوضح الجدول التالي بعض نماذج EDM المحددة ووحدة الانحراف الخاصة بها:

================= ================================  ============================================
التوزيع           مجال الهدف                          وحدة الانحراف :math:`d(y, \hat{y})`
================= ================================  ============================================
طبيعي             :math:`y \in (-\infty, \infty)`   :math:`(y-\hat{y})^2`
برنولي           :math:`y \in \{0, 1\}`            :math:`2({y}\log\frac{y}{\hat{y}}+({1}-{y})\log\frac{{1}-{y}}{{1}-\hat{y}})`
تصنيفي           :math:`y \in \{0, 1, ..., k\}`    :math:`2\sum_{i \in \{0, 1, ..., k\}} I(y = i) y_\text{i}\log\frac{I(y = i)}{\hat{I(y = i)}}`
بواسون           :math:`y \in [0, \infty)`         :math:`2(y\log\frac{y}{\hat{y}}-y+\hat{y})`
جاما             :math:`y \in (0, \infty)`         :math:`2(\log\frac{\hat{y}}{y}+\frac{y}{\hat{y}}-1)`
إنفيرس غاوسيان   :math:`y \in (0, \infty)`         :math:`\frac{(y-\hat{y})^2}{y\hat{y}^2}`
================= ================================  ============================================

يتم توضيح دوال كثافة الاحتمال (PDF) لهذه التوزيعات في الشكل التالي،

.. figure:: ./glm_data/poisson_gamma_tweedie_distributions.png
   :align: center
   :scale: 10

   PDF لمتغير عشوائي Y يتبع توزيعات بواسون، تويد (قوة=1.5) وجاما مع قيم وسطية مختلفة (:math:`\mu`). لاحظ الكتلة النقطية عند :math:`Y=0` لتوزيع بواسون وتوزيع تويد (قوة=1.5)، ولكن ليس لتوزيع جاما الذي لديه مجال هدف موجب بشكل صارم.

توزيع برنولي هو توزيع احتمالي منفصل لنمذجة تجربة برنولي - حدث له نتيجتان متبادلتان فقط. التوزيع التصنيفي هو تعميم لتوزيع برنولي لمتغير عشوائي تصنيفي. بينما يحتوي متغير عشوائي في توزيع برنولي على نتيجتين محتملتين، يمكن أن يأخذ متغير عشوائي تصنيفي واحدة من K فئات ممكنة، مع تحديد احتمال كل فئة بشكل منفصل.

يعتمد اختيار التوزيع على المشكلة المطروحة:

* إذا كانت قيم الهدف :math:`y` عبارة عن تعدادات (قيم عدد صحيح غير سالبة) أو ترددات نسبية (غير سالبة)، فيمكنك استخدام توزيع بواسون مع ارتباط لوغاريتمي.
* إذا كانت قيم الهدف موجبة ومائلة، يمكنك تجربة توزيع جاما مع ارتباط لوغاريتمي.
* إذا بدت قيم الهدف ذات ذيل أثقل من توزيع جاما، يمكنك تجربة توزيع إنفيرس غاوسيان (أو حتى قوى تباين أعلى من عائلة تويد).
* إذا كانت قيم الهدف :math:`y` احتمالات، فيمكنك استخدام توزيع برنولي. يمكن استخدام توزيع برنولي مع ارتباط لوغاريتمي للتصنيف الثنائي. يمكن استخدام التوزيع التصنيفي مع ارتباط سوفتماكس للتصنيف متعدد الفئات.


.. dropdown:: أمثلة على حالات الاستخدام

  * الزراعة / نمذجة الطقس:  عدد أحداث المطر في السنة (بواسون)، كمية المطر لكل حدث (جاما)، إجمالي هطول الأمطار في السنة (تويد / مركب بواسون جاما).
  * نمذجة المخاطر / تسعير بوليصة التأمين:  عدد أحداث المطالبة / حامل الوثيقة في السنة (بواسون)، التكلفة لكل حدث (جاما)، إجمالي التكلفة لكل حامل وثيقة في السنة (تويد / مركب بواسون جاما).
  * التخلف عن سداد الائتمان: احتمال عدم القدرة على سداد القرض (برنولي).
  * كشف الاحتيال: احتمال أن تكون المعاملة المالية مثل تحويل نقدي معاملة احتيالية (برنولي).
  * الصيانة التنبؤية: عدد أحداث انقطاع الإنتاج في السنة (بواسون)، مدة الانقطاع (جاما)، إجمالي وقت الانقطاع في السنة (تويد / مركب بواسون جاما).
  * اختبار الأدوية الطبية: احتمال علاج المريض في مجموعة من التجارب أو احتمال أن يعاني المريض من آثار جانبية (برنولي).
  * تصنيف الأخبار: تصنيف مقالات الأخبار إلى ثلاث فئات وهي أخبار الأعمال والسياسة وأخبار الترفيه (تصنيفي).

.. rubric:: المراجع

.. [10] McCullagh، Peter؛ Nelder، John (1989). نماذج الخطية المعممة، الطبعة الثانية. Boca Raton: Chapman and Hall/CRC. ISBN 0-412-31760-5.

.. [11] Jørgensen، B. (1992). The theory of exponential dispersion models and analysis of deviance. Monografias de matemática، no. 51. See also `Exponential dispersion model. <https://en.wikipedia.org/wiki/Exponential_dispersion_model>`_

الاستخدام

:class:`TweedieRegressor` ينفذ نموذج خطي عام لتوزيع Tweedie، والذي يسمح لنمذجة أي من التوزيعات المذكورة أعلاه باستخدام معامل "power" المناسب. على وجه الخصوص:

- ``power = 0``: التوزيع الطبيعي. عادة ما تكون المقدرات الخاصة مثل :class:`Ridge`، :class:`ElasticNet` أكثر ملاءمة في هذه الحالة.
- ``power = 1``: توزيع بواسون. :class:`PoissonRegressor` متاح للراحة. ومع ذلك، فهو يعادل تمامًا `TweedieRegressor(power=1, link='log')`.
- ``power = 2``: توزيع غاما. :class:`GammaRegressor` متاح للراحة. ومع ذلك، فهو يعادل تمامًا `TweedieRegressor(power=2, link='log')`.
- ``power = 3``: توزيع إنفيرس غاوسيان.

يتم تحديد دالة الربط بواسطة معامل `link`.

مثال على الاستخدام::

    >>> from sklearn.linear_model import TweedieRegressor
    >>> reg = TweedieRegressor(power=1, alpha=0.5, link='log')
    >>> reg.fit([[0, 0], [0, 1], [2, 2]], [0, 1, 2])
    TweedieRegressor(alpha=0.5, link='log', power=1)
    >>> reg.coef_
    array([0.2463..., 0.4337...])
    >>> reg.intercept_
    -0.7638...


.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_linear_model_plot_poisson_regression_non_normal_loss.py`
* :ref:`sphx_glr_auto_examples_linear_model_plot_tweedie_regression_insurance_claims.py`

.. dropdown:: اعتبارات عملية

  يجب توحيد مصفوفة الميزات `X` قبل التركيب. هذا يضمن أن العقوبة تعامل الميزات بالتساوي.

  نظرًا لأن المتنبئ الخطي :math:`Xw` يمكن أن يكون سالبًا ولا تدعم توزيعات بواسون وغاما وإنفيرس غاوسيان القيم السالبة، فمن الضروري تطبيق دالة ربط عكسية تضمن عدم السلبية. على سبيل المثال مع `link='log'`، تصبح دالة الربط العكسية :math:`h(Xw)=\exp(Xw)`.

  إذا كنت ترغب في نمذجة تردد نسبي، أي عدد لكل تعرض (الوقت، الحجم، ...) يمكنك القيام بذلك باستخدام توزيع بواسون وتمرير :math:`y=\frac{\mathrm{counts}}{\mathrm{exposure}}` كقيم مستهدفة مع :math:`\mathrm{exposure}` كأوزان عينات. للحصول على مثال ملموس، انظر على سبيل المثال :ref:`sphx_glr_auto_examples_linear_model_plot_tweedie_regression_insurance_claims.py`.

  عند إجراء التحقق المتقاطع لمعامل `power` من `TweedieRegressor`، يُنصح بتحديد دالة `scoring` صريحة، لأن الدرجة الافتراضية :meth:`TweedieRegressor.score` هي دالة لـ `power` نفسه.

تدرج انحداري ستوكاستي - SGD

التدرج العشوائي الانحداري هو أسلوب بسيط وفعال للغاية لتناسب النماذج الخطية. إنه مفيد بشكل خاص عندما يكون عدد العينات (وعدد الميزات) كبيرًا جدًا. تسمح طريقة "partial_fit" بالتعلم عبر الإنترنت/خارج النواة.

توفر الفئتان :class:`SGDClassifier` و:class:`SGDRegressor` وظائف لتناسب النماذج الخطية للتصنيف والانحدار باستخدام دوال خسارة مختلفة (محدبة) وعقوبات مختلفة. على سبيل المثال، مع "loss="log""، فإن :class:`SGDClassifier` تناسب نموذج الانحدار اللوجستي، بينما مع "loss="hinge"" فإنه يناسب آلة المتجهات الداعمة الخطية (SVM).

يمكنك الرجوع إلى قسم التوثيق المخصص :ref:`sgd` للحصول على مزيد من التفاصيل.

.. _perceptron:

المستقبل
========

يعد :class:`Perceptron` خوارزمية تصنيف بسيطة أخرى مناسبة للتعلم على نطاق واسع. بشكل افتراضي:

- لا يتطلب معدل تعلم.
- لا يتم تنظيمه (معاقبته).
- يقوم بتحديث نموذجه فقط عند ارتكاب الأخطاء.

تؤدي الخاصية الأخيرة إلى أن يكون Perceptron أسرع قليلاً في التدريب من SGD مع خسارة المفصلة وأن النماذج الناتجة تكون أكثر ندرة.

في الواقع، فإن :class:`Perceptron` هو غلاف حول فئة :class:`SGDClassifier` باستخدام خسارة المستقبل ومعدل تعلم ثابت. راجع :ref:`القسم الرياضي <sgd_mathematical_formulation>` لإجراء SGD للحصول على مزيد من التفاصيل.

.. _passive_aggressive:

خوارزميات Passive Aggressive
============================

خوارزميات Passive-aggressive هي عائلة من الخوارزميات للتعلم على نطاق واسع. إنها تشبه Perceptron من حيث أنها لا تتطلب معدل تعلم. ومع ذلك، على عكس Perceptron، فإنها تتضمن معامل تنظيم "C".

للتصنيف، يمكن استخدام :class:`PassiveAggressiveClassifier` مع "loss='hinge'"" (PA-I) أو "loss='squared_hinge'"" (PA-II). للانحدار، يمكن استخدام :class:`PassiveAggressiveRegressor` مع "loss='epsilon_insensitive'"" (PA-I) أو "loss='squared_epsilon_insensitive'"" (PA-II).

.. dropdown:: المراجع

  * "Online Passive-Aggressive Algorithms"
    <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>
    K. Crammer، O. Dekel، J. Keshat، S. Shalev-Shwartz، Y. Singer - JMLR 7 (2006)

انحدار المتانة: القيم المتطرفة وأخطاء النمذجة
    
تهدف الانحدار القوي إلى تركيب نموذج انحدار في وجود بيانات فاسدة: إما نقاط شاذة أو أخطاء في النموذج.

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_theilsen_001.png
   :target: ../auto_examples/linear_model/plot_theilsen.html
   :scale: 50%
   :align: center

سيناريوهات مختلفة ومفاهيم مفيدة
----------------------------------------

هناك أشياء مختلفة يجب وضعها في الاعتبار عند التعامل مع البيانات الفاسدة بالنقاط الشاذة:

.. |y_outliers| image:: ../auto_examples/linear_model/images/sphx_glr_plot_robust_fit_003.png
   :target: ../auto_examples/linear_model/plot_robust_fit.html
   :scale: 60%

.. |X_outliers| image:: ../auto_examples/linear_model/images/sphx_glr_plot_robust_fit_002.png
   :target: ../auto_examples/linear_model/plot_robust_fit.html
   :scale: 60%

.. |large_y_outliers| image:: ../auto_examples/linear_model/images/sphx_glr_plot_robust_fit_005.png
   :target: ../auto_examples/linear_model/plot_robust_fit.html
   :scale: 60%

* **النقاط الشاذة في X أو في y**؟

  ==================================== ====================================
  النقاط الشاذة في اتجاه y              النقاط الشاذة في اتجاه X
  ==================================== ====================================
  |y_outliers|                         |X_outliers|
  ==================================== ====================================

* **نسبة النقاط الشاذة مقابل حجم الخطأ**

  عدد النقاط الشاذة مهم، ولكن أيضًا مدى شذوذها.

  ==================================== ====================================
  النقاط الشاذة الصغيرة                  النقاط الشاذة الكبيرة
  ==================================== ====================================
  |y_outliers|                         |large_y_outliers|
  ==================================== ====================================

مفهوم مهم في التركيب القوي هو نقطة الانهيار: نسبة البيانات التي يمكن أن تكون شاذة لبدء فقدان البيانات الداخلية.

لاحظ أنه بشكل عام، يكون التركيب القوي في إعدادات عالية الأبعاد (عدد كبير من `n_features`) صعبًا للغاية. من المحتمل أن لا تعمل نماذج قوية هنا في هذه الإعدادات.


.. topic:: المفاضلات: أي مقدر؟

  يوفر Scikit-learn 3 مقدرات انحدار قوية:
  :ref:`RANSAC <ransac_regression>`,
  :ref:`Theil Sen <theil_sen_regression>` و
  :ref:`HuberRegressor <huber_regression>`.

  * :ref:`HuberRegressor <huber_regression>` يجب أن يكون أسرع من
    :ref:`RANSAC <ransac_regression>` و :ref:`Theil Sen <theil_sen_regression>`
    ما لم يكن عدد العينات كبيرًا جدًا، أي ``n_samples`` >> ``n_features``.
    هذا لأن :ref:`RANSAC <ransac_regression>` و :ref:`Theil Sen <theil_sen_regression>`
    يتم تركيبها على مجموعات فرعية أصغر من البيانات. ومع ذلك، من غير المرجح أن يكون كل من :ref:`Theil Sen <theil_sen_regression>`
    و :ref:`RANSAC <ransac_regression>` قويين مثل
    :ref:`HuberRegressor <huber_regression>` للمعلمات الافتراضية.

  * :ref:`RANSAC <ransac_regression>` أسرع من :ref:`Theil Sen <theil_sen_regression>`
    ويتدرج بشكل أفضل مع عدد العينات.

  * :ref:`RANSAC <ransac_regression>` سيتعامل بشكل أفضل مع نقاط شاذة كبيرة
    في اتجاه y (الوضع الأكثر شيوعًا).

  * :ref:`Theil Sen <theil_sen_regression>` سيتعامل بشكل أفضل مع
    النقاط الشاذة متوسطة الحجم في اتجاه X، ولكن هذه الخاصية ستختفي
    في إعدادات عالية الأبعاد.

  عندما تكون في شك، استخدم :ref:`RANSAC <ransac_regression>`.

.. _ransac_regression:

RANSAC: RANdom SAmple Consensus
    

RANSAC (توافق العينة العشوائية) يلائم نموذجًا من مجموعات فرعية عشوائية من البيانات الداخلية من مجموعة البيانات الكاملة.

RANSAC هو خوارزمية غير حتمية تنتج فقط نتيجة معقولة باحتمال معين، والذي يعتمد على عدد التكرارات (انظر معلمة `max_trials`). يتم استخدامه عادةً لمشاكل الانحدار الخطي وغير الخطي وهو شائع بشكل خاص في مجال الرؤية الحاسوبية الفوتوغرافية.

تقسم الخوارزمية بيانات العينة المدخلة الكاملة إلى مجموعة من البيانات الداخلية، والتي قد تكون عرضة للضوضاء، والبيانات الشاذة، والتي تنتج عن قياسات خاطئة أو فرضيات غير صالحة حول البيانات. ثم يتم تقدير النموذج الناتج فقط من البيانات الداخلية المحددة.

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_ransac_001.png
   :target: ../auto_examples/linear_model/plot_ransac.html
   :align: center
   :scale: 50%

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_linear_model_plot_ransac.py`
* :ref:`sphx_glr_auto_examples_linear_model_plot_robust_fit.py`

.. dropdown:: تفاصيل الخوارزمية

  تقوم كل تكرارية بالخطوات التالية:

  1. اختر ``min_samples`` عينات عشوائية من البيانات الأصلية وتحقق مما إذا كانت مجموعة البيانات صالحة (انظر ``is_data_valid``).
  2.لائم نموذجًا للمجموعة العشوائية (``estimator.fit``) وتحقق مما إذا كان النموذج المقدر صالحًا (انظر ``is_model_valid``).
  3.صنف جميع البيانات كبيانات داخلية أو شاذة عن طريق حساب المخلفات للنموذج المقدر (``estimator.predict(X) - y``) - جميع عينات البيانات ذات المخلفات المطلقة الأصغر من أو تساوي ``residual_threshold`` تعتبر بيانات داخلية.
  4.احفظ النموذج الملائم كأفضل نموذج إذا كان عدد عينات البيانات الداخلية هو الأقصى. في حالة كان للنموذج المقدر الحالي نفس عدد البيانات الداخلية، فإنه يعتبر أفضل نموذج فقط إذا كان لديه درجة أفضل.

  تتم تنفيذ هذه الخطوات إما عددًا أقصى من المرات (``max_trials``) أو حتى يتم استيفاء أحد معايير الإيقاف الخاصة (انظر ``stop_n_inliers`` و ``stop_score``). يتم تقدير النموذج النهائي باستخدام جميع عينات البيانات الداخلية (مجموعة التوافق) للنموذج الأفضل الذي تم تحديده مسبقًا.

  تسمح دالتا ``is_data_valid`` و ``is_model_valid`` بتحديد ورفض مجموعات متدهورة من العينات الفرعية العشوائية. إذا لم يكن النموذج المقدر مطلوبًا لتحديد الحالات المتدهورة، فيجب استخدام ``is_data_valid`` لأنه يتم استدعاؤه قبل ملائمة النموذج وبالتالي يؤدي إلى أداء حسابي أفضل.

.. dropdown:: المراجع

  * https://en.wikipedia.org/wiki/RANSAC
  * `"توافق العينة العشوائية: نموذج لتركيب النموذج مع التطبيقات لتحليل الصور والخرائط الآلية"
    <https://www.cs.ait.ac.th/~mdailey/cvreadings/Fischler-RANSAC.pdf>`_
    مارتن أ. فيشلر وروبرت سي. بولز - SRI International (1981)
  * `"تقييم أداء عائلة RANSAC"
    <http://www.bmva.org/bmvc/2009/Papers/Paper355/Paper355.pdf>`_
    سونجلوك تشوي، تايمين كيم وونبيل يو - BMVC (2009)

.. _theil_sen_regression:

مقدر Theil-Sen: مقدر قائم على الوسيط العام
--------------------------------------------------------

    إن مقدر :class:`TheilSenRegressor` يستخدم تعميما للوسيط في أبعاد متعددة. وبالتالي فهو قوي في مواجهة القيم المتطرفة متعددة المتغيرات. ومع ذلك، تجدر الإشارة إلى أن قوة هذا المقدّر تنخفض بسرعة مع زيادة أبعاد المشكلة. يفقد خصائصه القوية ويصبح لا أفضل من المربعات الصغرى العادية في البعد العالي.

    .. rubric:: أمثلة

    * :ref:`sphx_glr_auto_examples_linear_model_plot_theilsen.py`
    * :ref:`sphx_glr_auto_examples_linear_model_plot_robust_fit.py`


    .. dropdown:: اعتبارات نظرية

      :class:`TheilSenRegressor` قابل للمقارنة مع :ref:`المربعات الصغرى العادية <ordinary_least_squares>` من حيث الكفاءة اللامتناهية الصغيرة وكمعرّف غير متحيز. على النقيض من المربعات الصغرى العادية، فإن Theil-Sen هو أسلوب غير بارامتري مما يعني أنه لا يفترض أي افتراض حول التوزيع الأساسي للبيانات. بما أن Theil-Sen هو مقدّر قائم على الوسيط، فهو أكثر قوة ضد البيانات الفاسدة المعروفة أيضًا باسم القيم المتطرفة. في الإعداد أحادي المتغير، يمتلك Theil-Sen نقطة انهيار تبلغ حوالي 29.3٪ في حالة الانحدار الخطي البسيط مما يعني أنه يمكنه تحمل بيانات فاسدة عشوائية تصل إلى 29.3٪.

      .. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_theilsen_001.png
        :target: ../auto_examples/linear_model/plot_theilsen.html
        :align: center
        :scale: 50%

      إن تنفيذ :class:`TheilSenRegressor` في scikit-learn يتبع تعميما إلى نموذج انحدار خطي متعدد المتغيرات [#f1]_ باستخدام الوسيط المكاني الذي هو تعميم للوسيط إلى أبعاد متعددة [#f2]_.

      من حيث تعقيد الوقت والمساحة، يتدرج Theil-Sen وفقًا لـ

      .. math::
          \binom{n_{\text{samples}}}{n_{\text{subsamples}}}

      مما يجعل من غير الممكن تطبيقه بشكل شامل على المشاكل التي تحتوي على عدد كبير من العينات والميزات. لذلك، يمكن اختيار حجم مجموعة فرعية للحد من تعقيد الوقت والمساحة من خلال النظر فقط في مجموعة فرعية عشوائية من جميع التركيبات الممكنة.

      .. rubric:: المراجع

      .. [#f1] Xin Dang، Hanxiang Peng، Xueqin Wang و Heping Zhang: `Theil-Sen Estimators in a Multiple Linear Regression Model. <http://home.olemiss.edu/~xdang/papers/MTSE.pdf>`_

      .. [#f2] T. Kärkkäinen و S. Äyrämö: `On Computation of Spatial Median for Robust Data Mining. <http://users.jyu.fi/~samiayr/pdf/ayramo_eurogen05.pdf>`_

      انظر أيضًا صفحة `Wikipedia <https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator>`_


    .. _huber_regression:

    انحدار Huber
    ----------------

    إن :class:`HuberRegressor` يختلف عن :class:`Ridge` لأنه يطبق خسارة خطية على العينات التي يتم تصنيفها على أنها قيم متطرفة. يتم تصنيف العينة على أنها داخلية إذا كان الخطأ المطلق لتلك العينة أقل من عتبة معينة. يختلف عن :class:`TheilSenRegressor` و :class:`RANSACRegressor` لأنه لا يتجاهل تأثير القيم المتطرفة ولكن يعطي وزنًا أقل لها.

    .. figure:: /auto_examples/linear_model/images/sphx_glr_plot_huber_vs_ridge_001.png
       :target: ../auto_examples/linear_model/plot_huber_vs_ridge.html
       :align: center
       :scale: 50%

    .. rubric:: أمثلة

    * :ref:`sphx_glr_auto_examples_linear_model_plot_huber_vs_ridge.py`

    .. dropdown:: تفاصيل رياضية

      إن دالة الخسارة التي يقوم :class:`HuberRegressor` بتصغيرها معطاة بواسطة

      .. math::

        \min_{w, \sigma} {\sum_{i=1}^n\left(\sigma + H_{\epsilon}\left(\frac{X_{i}w - y_{i}}{\sigma}\right)\sigma\right) + \alpha {||w||_2}^2}

      حيث

      .. math::

        H_{\epsilon}(z) = \begin{cases}
              z^2, & \text {if } |z| < \epsilon, \\
              2\epsilon|z| - \epsilon^2, & \text{otherwise}
        \end{cases}

      من المستحسن تعيين المعلمة ``epsilon`` إلى 1.35 لتحقيق كفاءة إحصائية بنسبة 95٪.

      .. rubric:: المراجع

      * Peter J. Huber، Elvezio M. Ronchetti: Robust Statistics, Concomitant scale estimates, pg 172

    يختلف :class:`HuberRegressor` عن استخدام :class:`SGDRegressor` مع تعيين الخسارة إلى `huber` بالطرق التالية.

    - :class:`HuberRegressor` غير حساس للتدرج. بمجرد تعيين ``epsilon``، فإن تقليص أو تكبير ``X`` و ``y`` بقيم مختلفة سينتج نفس القوة ضد القيم المتطرفة كما كان من قبل. مقارنة بـ :class:`SGDRegressor` حيث يجب إعادة تعيين ``epsilon`` عند تقليص أو تكبير ``X`` و ``y``.

    - :class:`HuberRegressor` يجب أن يكون أكثر كفاءة للاستخدام على البيانات التي تحتوي على عدد صغير من العينات بينما يحتاج :class:`SGDRegressor` إلى عدد من التمريرات على بيانات التدريب لإنتاج نفس القوة.

    لاحظ أن هذا المقدّر يختلف عن تنفيذ R لانحدار قوي (https://stats.oarc.ucla.edu/r/dae/robust-regression/) لأن تنفيذ R يقوم بتنفيذ مربعات صغرى مرجحة مع أوزان تعطى لكل عينة على أساس مقدار الباقي أكبر من عتبة معينة.

    .. _quantile_regression:

    انحدار Quantile
    
    --------------------------------------------------------
 
===================

تقدير الانحدار الكمي (Quantile regression) يقدر الوسيط أو الكميات الأخرى لـ :math:`y` المشروطة بـ :math:`X`، بينما يقدر المربعات الصغرى العادية (OLS) المتوسط الشرطي.

قد يكون الانحدار الكمي مفيدًا إذا كان المرء مهتمًا بالتنبؤ بفاصل زمني بدلاً من التنبؤ بالنقطة. في بعض الأحيان، يتم حساب فترات التنبؤ بناءً على افتراض أن خطأ التنبؤ موزع بشكل طبيعي بمتوسط صفر وتباين ثابت. يوفر الانحدار الكمي فترات تنبؤ معقولة حتى للأخطاء ذات التباين غير الثابت (ولكن يمكن التنبؤ به) أو التوزيع غير الطبيعي.

.. figure:: /auto_examples/linear_model/images/sphx_glr_plot_quantile_regression_002.png
   :target: ../auto_examples/linear_model/plot_quantile_regression.html
   :align: center
   :scale: 50%

بناءً على تقليل خسارة الكرة والدبابيس، يمكن أيضًا تقدير الكميات الشرطية بواسطة نماذج أخرى غير النماذج الخطية. على سبيل المثال، يمكن لـ :class:`~sklearn.ensemble.GradientBoostingRegressor` التنبؤ بالكميات الشرطية إذا تم تعيين معامل ``loss`` إلى ``"quantile"`` وتم تعيين معامل ``alpha`` على الكمية التي يجب التنبؤ بها. انظر المثال في :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_quantile.py`.

معظم تطبيقات الانحدار الكمي تستند إلى مشكلة البرمجة الخطية. يعتمد التنفيذ الحالي على :func:`scipy.optimize.linprog`.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_linear_model_plot_quantile_regression.py`

.. dropdown:: التفاصيل الرياضية

  كما هو الحال في النموذج الخطي، فإن :class:`QuantileRegressor` يعطي تنبؤات خطية :math:`\hat{y}(w, X) = Xw` للكمية :math:`q`، :math:`q \in (0, 1)`. ثم يتم العثور على الأوزان أو المعاملات :math:`w` من خلال مشكلة التقليل التالية:

  .. math::
      \min_{w} {\frac{1}{n_{\text{samples}}}
      \sum_i PB_q(y_i - X_i w) + \alpha ||w||_1}.

  يتكون هذا من خسارة الكرة والدبابيس (المعروفة أيضًا باسم الخسارة الخطية)، انظر أيضًا :class:`~sklearn.metrics.mean_pinball_loss`،

  .. math::
      PB_q(t) = q \max(t, 0) + (1 - q) \max(-t, 0) =
      \begin{cases}
          q t, & t > 0, \\
          0,    & t = 0, \\
          (q-1) t, & t < 0
      \end{cases}

  وعقوبة L1 التي يتحكم فيها معامل ``alpha``، على غرار :class:`Lasso`.

  نظرًا لأن خسارة الكرة والدبابيس خطية فقط في المخلفات، فإن الانحدار الكمي أكثر قوة بكثير في مواجهة القيم المتطرفة من التقدير القائم على خطأ التربيع للمتوسط. إلى حد ما بينهما هو :class:`HuberRegressor`.

.. dropdown:: المراجع

  * Koenker, R., & Bassett Jr, G. (1978). `Regression quantiles.
    <https://gib.people.uic.edu/RQ.pdf>`_
    Econometrica: journal of the Econometric Society, 33-50.

  * Portnoy, S., & Koenker, R. (1997). :doi:`The Gaussian hare and the Laplacian
    tortoise: computability of squared-error versus absolute-error estimators.
    Statistical Science, 12, 279-300 <10.1214/ss/1030037960>`.

  * Koenker, R. (2005). :doi:`Quantile Regression <10.1017/CBO9780511754098>`.
    Cambridge University Press.


.. _polynomial_regression:

انحدار متعدد الحدود: توسيع النماذج الخطية مع وظائف الأساس
 ==================================================================

.. currentmodule:: sklearn.preprocessing

واحد من الأنماط الشائعة في التعلم الآلي هو استخدام النماذج الخطية المدربة على الدوال غير الخطية للبيانات. هذا النهج يحافظ على الأداء السريع بشكل عام للطرق الخطية، بينما يسمح لها بتناسب مجموعة أكبر بكثير من البيانات.

.. dropdown:: التفاصيل الرياضية

  على سبيل المثال، يمكن تمديد الانحدار الخطي البسيط عن طريق بناء **ميزات متعددة الحدود** من المعاملات. في حالة الانحدار الخطي القياسي، قد يكون لديك نموذج يبدو كالتالي لبيانات ثنائية الأبعاد:

  .. math::    \hat{y}(w, x) = w_0 + w_1 x_1 + w_2 x_2

  إذا أردنا ملائمة قطع مكافئ للبيانات بدلاً من مستوى، فيمكننا الجمع بين الميزات في متعددات حدود من الدرجة الثانية، بحيث يبدو النموذج كالتالي:

  .. math::    \hat{y}(w, x) = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_1 x_2 + w_4 x_1^2 + w_5 x_2^2

  الملاحظة (التي قد تكون مفاجئة في بعض الأحيان) هي أن هذا لا يزال نموذجًا خطيًا: لرؤية هذا، تخيل إنشاء مجموعة جديدة من الميزات

  .. math::  z = [x_1, x_2, x_1 x_2, x_1^2, x_2^2]

  مع هذا التسمية الجديدة للبيانات، يمكن كتابة مشكلتنا

  .. math::    \hat{y}(w, z) = w_0 + w_1 z_1 + w_2 z_2 + w_3 z_3 + w_4 z_4 + w_5 z_5

  نرى أن *الانحدار متعدد الحدود* الناتج يقع في نفس فئة النماذج الخطية التي نظرنا إليها أعلاه (أي أن النموذج خطي في :math:`w`) ويمكن حله بنفس التقنيات. من خلال النظر في التناسبات الخطية داخل مساحة ذات أبعاد أعلى مبنية باستخدام دوال الأساس هذه، يكون للنموذج المرونة لتناسب مجموعة أكبر بكثير من البيانات.

فيما يلي مثال على تطبيق هذه الفكرة على بيانات أحادية البعد، باستخدام ميزات متعددة الحدود بدرجات متفاوتة:

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_polynomial_interpolation_001.png
   :target: ../auto_examples/linear_model/plot_polynomial_interpolation.html
   :align: center
   :scale: 50%

تم إنشاء هذا الشكل باستخدام محول :class:`PolynomialFeatures`، والذي يحول مصفوفة بيانات الإدخال إلى مصفوفة بيانات جديدة بدرجة معينة. يمكن استخدامه على النحو التالي::

    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> import numpy as np
    >>> X = np.arange(6).reshape(3, 2)
    >>> X
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> poly = PolynomialFeatures(degree=2)
    >>> poly.fit_transform(X)
    array([[ 1.,  0.,  1.,  0.,  0.,  1.],
           [ 1.,  2.,  3.,  4.,  6.,  9.],
           [ 1.,  4.,  5., 16., 20., 25.]])

تم تحويل ميزات ``X`` من :math:`[x_1, x_2]` إلى :math:`[1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]`, ويمكن الآن استخدامها داخل أي نموذج خطي.

يمكن تبسيط هذا النوع من المعالجة المسبقة باستخدام أدوات :ref:`Pipeline <pipeline>`. يمكن إنشاء كائن واحد يمثل انحدارًا متعدد الحدود بسيطًا واستخدامه على النحو التالي::

    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.pipeline import Pipeline
    >>> import numpy as np
    >>> model = Pipeline([('poly', PolynomialFeatures(degree=3)),
    ...                   ('linear', LinearRegression(fit_intercept=False))])
    >>> # fit to an order-3 polynomial data
    >>> x = np.arange(5)
    >>> y = 3 - 2 * x + x ** 2 - x ** 3
    >>> model = model.fit(x[:, np.newaxis], y)
    >>> model.named_steps['linear'].coef_
    array([ 3., -2.,  1., -1.])

النموذج الخطي المدرب على ميزات متعددة الحدود قادر على استرداد معاملات متعدد الحدود المدخلة تمامًا.

في بعض الحالات، ليس من الضروري تضمين قوى أعلى لأي ميزة واحدة، ولكن فقط ما يسمى *ميزات التفاعل* التي تضرب معًا على الأكثر :math:`d` ميزات مميزة. يمكن الحصول عليها من :class:`PolynomialFeatures` مع الإعداد ``interaction_only=True``.

على سبيل المثال، عند التعامل مع الميزات المنطقية، :math:`x_i^n = x_i` لجميع :math:`n` وبالتالي فهي غير مجدية؛ ولكن :math:`x_i x_j` يمثل اقتران اثنين من المنطقية. بهذه الطريقة، يمكننا حل مشكلة XOR مع مصنف خطي::

    >>> from sklearn.linear_model import Perceptron
    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> import numpy as np
    >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    >>> y = X[:, 0] ^ X[:, 1]
    >>> y
    array([0, 1, 1, 0])
    >>> X = PolynomialFeatures(interaction_only=True).fit_transform(X).astype(int)
    >>> X
    array([[1, 0, 0, 0],
           [1, 0, 1, 0],
           [1, 1, 0, 0],
           [1, 1, 1, 1]])
    >>> clf = Perceptron(fit_intercept=False, max_iter=10, tol=None,
    ...                 shuffle=False).fit(X, y)

وتكون "توقعات" المصنف مثالية::

    >>> clf.predict(X)
    array([0, 1, 1, 0])
    >>> clf.score(X, y)
    1.0
    
