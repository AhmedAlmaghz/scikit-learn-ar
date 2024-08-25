    .. _kernel_ridge:

    ===========================
    انحدار ريدج باستخدام النواة
    ===========================

    .. currentmodule:: sklearn.kernel_ridge

    يجمع انحدار ريدج باستخدام النواة (KRR) [M2012]_ بين :ref:`ridge_regression` (أقل مربعات خطية مع تنظيم هنجلроين باستخدام معيار l2) و "خدعة النواة" (<https://en.wikipedia.org/wiki/Kernel_method>_). وبالتالي، فإنه يتعلم دالة خطية في الفضاء المستحث بواسطة النواة والبيانات المعنية. بالنسبة للأنوية غير الخطية، فإن هذا يتوافق مع دالة غير خطية في الفضاء الأصلي.

    شكل النموذج الذي يتعلمه :class:`KernelRidge` مطابق لانحدار المتجه الداعم (:class:`~sklearn.svm.SVR`). ومع ذلك، يتم استخدام دالات خسارة مختلفة: يستخدم KRR خسارة الخطأ المربع بينما يستخدم انحدار المتجه الداعم خسارة غير حساسة لـ :math:`\epsilon`، وكلاهما يجمع مع تنظيم l2. على النقيض من :class:`~sklearn.svm.SVR`، يمكن إجراء تركيب :class:`KernelRidge` بشكل مغلق وعادة ما يكون أسرع للمجموعات البيانات متوسطة الحجم. من ناحية أخرى، يكون النموذج المتعلم غير متفرق وبالتالي أبطأ من :class:`~sklearn.svm.SVR`، الذي يتعلم نموذجًا متفرقًا لـ :math:`\epsilon > 0`، في وقت التنبؤ.

    تقارن الصورة التالية بين :class:`KernelRidge` و :class:`~sklearn.svm.SVR` على مجموعة بيانات اصطناعية، والتي تتكون من دالة جيبية الهدف وضوضاء قوية تضاف إلى كل خامس نقطة بيانات. يتم رسم النموذج المتعلم من :class:`KernelRidge` و :class:`~sklearn.svm.SVR`، حيث تم تحسين كل من التعقيد/التنظيم وعرض النطاق الترددي لنواة RBF باستخدام البحث الشبكي. الدوال المتعلمة متشابهة جدًا؛ ومع ذلك، فإن تركيب :class:`KernelRidge` أسرع بحوالي سبع مرات من تركيب :class:`~sklearn.svm.SVR` (كلاهما مع البحث الشبكي). ومع ذلك، فإن التنبؤ بـ 100000 قيمة مستهدفة يكون أسرع بأكثر من ثلاث مرات مع :class:`~sklearn.svm.SVR` لأنه تعلم نموذجًا متفرقًا باستخدام حوالي 1/3 فقط من نقاط بيانات التدريب البالغ عددها 100 كنواقل دعم.

    .. figure:: ../auto_examples/miscellaneous/images/sphx_glr_plot_kernel_ridge_regression_001.png
       :target: ../auto_examples/miscellaneous/plot_kernel_ridge_regression.html
       :align: center

    تقارن الصورة التالية الوقت اللازم لتركيب والتنبؤ بـ :class:`KernelRidge` و :class:`~sklearn.svm.SVR` لأحجام مختلفة لمجموعة التدريب. يكون تركيب :class:`KernelRidge` أسرع من :class:`~sklearn.svm.SVR` لمجموعات بيانات التدريب متوسطة الحجم (أقل من 1000 عينة)؛ ومع ذلك، بالنسبة لمجموعات بيانات التدريب الأكبر، فإن :class:`~sklearn.svm.SVR` يتدرج بشكل أفضل. فيما يتعلق بوقت التنبؤ، يكون :class:`~sklearn.svm.SVR` أسرع من :class:`KernelRidge` لجميع أحجام مجموعة التدريب بسبب الحل المتفرق المتعلم. لاحظ أن درجة التفرق وبالتالي وقت التنبؤ يعتمد على معاملات :math:`\epsilon` و :math:`C` لـ :class:`~sklearn.svm.SVR`؛ :math:`\epsilon = 0` سيوافق نموذجًا كثيفًا.

    .. figure:: ../auto_examples/miscellaneous/images/sphx_glr_plot_kernel_ridge_regression_002.png
       :target: ../auto_examples/miscellaneous/plot_kernel_ridge_regression.html
       :align: center

    .. rubric:: أمثلة

    * :ref:`sphx_glr_auto_examples_miscellaneous_plot_kernel_ridge_regression.py`

    .. rubric:: المراجع

    .. [M2012] "التعلم الآلي: منظور احتمالي"
       ميرفي، ك. ب. - الفصل 14.4.3، الصفحات 492-493، مطبعة معهد ماساتشوستس للتكنولوجيا، 2012
