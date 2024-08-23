هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية، مع مراعاة عدم ترجمة الرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية:

.. _cython:

أفضل ممارسات وسوابق ومعرفة سيTHON

(ملاحظة: كلمة "Cython" هي اسم خاص بلغة البرمجة ولا تحتاج إلى ترجمة)

هذه الوثيقة تقدم نصائح لتطوير كود Cython في scikit-learn.

نصائح لتطوير الكود باستخدام Cython في scikit-learn

(ملاحظة: تم الحفاظ على الرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية دون ترجمة كما هو مطلوب)
================================================

نصائح لتسهيل التطوير
^^^^^^^^^^^^^^^^^^^^

* الوقت الذي تقضيه في قراءة `توثيق Cython <https://cython.readthedocs.io/en/latest/>`_ ليس وقتًا ضائعًا.

* إذا كنت تنوي استخدام OpenMP: في MacOS، لا يدعم توزيع النظام لـ ``clang`` OpenMP. يمكنك تثبيت حزمة ``compilers`` المتوفرة على ``conda-forge`` والتي تأتي مع تنفيذ OpenMP.

* قد يساعد تفعيل `التحقق <https://github.com/scikit-learn/scikit-learn/blob/62a017efa047e9581ae7df8bbaa62cf4c0544ee4/sklearn/_build_utils/__init__.py#L68-L87>`_. على سبيل المثال، لتفعيل boundscheck استخدم:

  .. code-block:: bash

         export SKLEARN_ENABLE_DEBUG_CYTHON_DIRECTIVES=1

* `ابدأ من الصفر في دفتر ملاحظات <https://cython.readthedocs.io/en/latest/src/quickstart/build.html#using-the-jupyter-notebook>`_ لفهم كيفية استخدام Cython والحصول على تعليقات حول عملك بسرعة. إذا كنت تخطط لاستخدام OpenMP في عمليات التنفيذ الخاصة بك في دفتر ملاحظات Jupyter الخاص بك، فقم بإضافة وسيطات إضافية للمترجم والرابط في سحر Cython.

  .. code-block:: python

         # For GCC and for clang
         %%cython --compile-args=-fopenmp --link-args=-fopenmp
         # For Microsoft's compilers
         %%cython --compile-args=/openmp --link-args=/openmp

* لاستكشاف أخطاءコード C (مثل الانهيار)، استخدم ``gdb`` مع:

  .. code-block:: bash

         gdb --ex r --args python ./entrypoint_to_bug_reproducer.py

* للوصول إلى بعض القيمة في مكان ما لاستكشاف الأخطاء في سياق ``cdef (nogil)``، استخدم:

  .. code-block:: cython

         with gil:
             print(state_to_print)

* لاحظ أن Cython لا يمكنه تحليل سلاسل f-strings مع تعبيرات ``{var=}``، على سبيل المثال

  .. code-block:: bash

         print(f"{test_val=}")

* تحتوي قاعدة كود scikit-learn على الكثير من أنواع غير موحدة (fused) (إعادة)التعريفات. هناك حاليًا `عمل مستمر لتبسيط ذلك وتوحيده عبر قاعدة الكود
  <https://github.com/scikit-learn/scikit-learn/issues/25572>`_. في الوقت الحالي، تأكد من فهمك لأنواع الخرسانة المستخدمة في نهاية المطاف.

* قد تجد هذا الاسم المستعار لتجميع ملحقات Cython الفردية في متناول اليد:

  .. code-block::

      # You might want to add this alias to your shell script config.
      alias cythonX="cython -X language_level=3 -X boundscheck=False -X wraparound=False -X initializedcheck=False -X nonecheck=False -X cdivision=True"

      # This generates `source.c` as if you had recompiled scikit-learn entirely.
      cythonX --annotate source.pyx

* استخدام الخيار ``--annotate`` مع هذا العلم يسمح بإنشاء تقرير HTML عن تعليق التعليمات البرمجية. يشير هذا التقرير إلى التفاعلات مع مترجم CPython على أساس كل سطر على حدة. يجب تجنب التفاعلات مع مترجم CPython قدر الإمكان في الأقسام الحسابية المكثفة للخوارزميات. لمزيد من المعلومات، يرجى الرجوع إلى `هذا القسم من البرنامج التعليمي لـ Cython <https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html#primes>`_

  .. code-block::

      # هذا يولد تقرير HTML (`source.html`) لـ `source.c`.
      cythonX --annotate source.pyx

نصائح للأداء
^^^^^^^^^^^^

* فهم GIL في سياق CPython (ما هي المشاكل التي يحلها، ما هي قيوده)
  وتحصل على فهم جيد لمتى سيتم تعيين Cython إلى كود C خالٍ من التفاعلات مع
  CPython، ومتى لن يكون، ومتى لا يمكن (على سبيل المثال، وجود تفاعلات مع Python
  الكائنات، والتي تشمل الوظائف). في هذا الصدد، يوفر `PEP073 <https://peps.python.org/pep-0703/>`_
  نظرة عامة جيدة وسياق وطرق لإزالة.

* تأكد من قيامك بتعطيل `التحقق <https://github.com/scikit-learn/scikit-learn/blob/62a017efa047e9581ae7df8bbaa62cf4c0544ee4/sklearn/_build_utils/__init__.py#L68-L87>`_.

* دائما تفضل memoryviews بدلا من ``cnp.ndarray`` عندما يكون ذلك ممكنا: memoryviews خفيفة الوزن.

* تجنب شريحة memoryview: قد تكون شريحة memoryview مكلفة أو مضللة في بعض الحالات ونفضل عدم استخدامها، حتى إذا كان التعامل مع أبعاد أقل في بعض السياقات سيكون أفضل.

* قم بتزيين الفئات أو الأساليب النهائية بـ ``@final`` (هذا يسمح بإزالة الجداول الافتراضية عند الحاجة)

* أساليب ودالات مضمنة عند الاقتضاء

* في الشك، قراءة التعليمات البرمجية C أو C ++ التي تم إنشاؤها إذا كنت تستطيع: "كلما قل عدد تعليمات C والتوجيهات لسطر من التعليمات البرمجية Cython، كان ذلك أفضل" هي قاعدة إرشادية جيدة.

* إعلانات ``nogil`` هي مجرد تلميحات: عند إعلان وظائف ``cdef``
  كما nogil، فهذا يعني أنه يمكن استدعاؤها دون عقد GIL، ولكنها لا تطلق
  GIL عند دخولهم. عليك أن تفعل ذلك بنفسك إما عن طريق تمرير ``nogil=True`` إلى
  ``cython.parallel.prange`` صراحة، أو باستخدام مدير سياق صريح:

  .. code-block:: cython

      cdef inline void my_func(self) nogil:

          # Some logic interacting with CPython, e.g. allocating arrays via NumPy.

          with nogil:
              # The code here is run as is it were written in C.

          return 0

  هذا البند يعتمد على `هذا التعليق من Stéfan's Benhel <https://github.com/cython/cython/issues/2798#issuecomment-459971828>`_

* المكالمات المباشرة إلى روتين BLAS ممكنة عبر واجهات محددة في ``sklearn.utils._cython_blas``.

استخدام OpenMP
^^^^^^^^^^^^^^

نظرًا لأنه يمكن بناء scikit-learn بدون OpenMP، فمن الضروري حماية كل
استدعاء مباشر لـ OpenMP.

يوفر وحدة `_openmp_helpers`، المتاحة في
`sklearn/utils/_openmp_helpers.pyx <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_openmp_helpers.pyx>`_
يوفر إصدارات محمية من إجراءات OpenMP. لاستخدام إجراءات OpenMP، يجب أن تكون ``cimported`` من هذه الوحدة النمطية وليس من مكتبة OpenMP مباشرة:

.. code-block:: cython

   from sklearn.utils._openmp_helpers cimport omp_get_max_threads
   max_threads = omp_get_max_threads()


حلقة متوازية، `prange`، محمية بالفعل بواسطة cython ويمكن استخدامها مباشرة
من `cython.parallel`.

أنواع
~~~~~
    
    

هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية، مع مراعاة عدم ترجمة الرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية:

يتطلب شفرة Cython استخدام أنواع البيانات الصريحة. وهذا أحد أسباب الحصول على تحسين في الأداء. ولتجنب ازدواجية الشفرة، لدينا مكان مركزي لأكثر الأنواع استخدامًا في `sklearn/utils/_typedefs.pyd <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_typedefs.pyd>`_. من الأفضل أن تبدأ بالنظر هناك و `cimport` الأنواع التي تحتاجها، على سبيل المثال:

.. code-block:: cython

    from sklear.utils._typedefs cimport float32, float64
    
    
(ملاحظة: تم الحفاظ على الشفرة البرمجية ورابط الملف كما هي في النص الأصلي دون ترجمة.)
