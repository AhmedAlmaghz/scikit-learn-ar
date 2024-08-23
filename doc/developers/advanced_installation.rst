هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية، مع الحفاظ على الرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية:

.. _advanced-installation:

.. include:: ../min_dependency_substitutions.rst

==================================================
تثبيت إصدار التطوير من scikit-learn
==================================================

يقدم هذا القسم شرحًا عن كيفية تثبيت **الفرع الرئيسي** من scikit-learn. يمكن القيام بذلك إما عن طريق تثبيت بناء ليلي أو بناء من المصدر.

.. _install_nightly_builds:

تثبيت البناء الليلي
=========================

تقوم خوادم التكامل المستمر لمشروع scikit-learn ببناء واختبار وتحميل حزم wheel للإصدار الأحدث من Python على أساس ليلي.

يعد تثبيت البناء الليلي أسرع طريقة لـ:

- تجربة ميزة جديدة سيتم إصدارها في الإصدار التالي (أي ميزة من طلب سحب تمت إضافته مؤخرًا إلى الفرع الرئيسي)؛

- التحقق مما إذا كان الخطأ الذي واجهته قد تم إصلاحه منذ الإصدار الأخير.

يمكنك تثبيت البناء الليلي لـ scikit-learn باستخدام الفهرس `scientific-python-nightly-wheels` من سجل PyPI لـ `anaconda.org`:

.. prompt:: bash $

 pip install --pre --extra-index https://pypi.anaconda.org/scientific-python-nightly-wheels/simple scikit-learn

لاحظ أنه قد يكون من الضروري إلغاء تثبيت scikit-learn أولاً لتتمكن من تثبيت البناء الليلي لـ scikit-learn.

.. _install_bleeding_edge:

البناء من المصدر

    

بناء المشروع من المصدر مطلوب للعمل على مساهمة (إصلاح الخلل، ميزة جديدة، تحسين الشفرة أو الوثائق).

.. _git_repo:

#. استخدم Git للتحقق من آخر مصدر من مستودع scikit-learn على Github.:

   .. prompt:: bash $

     git clone git@github.com:scikit-learn/scikit-learn.git  # أضف --depth 1 إذا كان الاتصال الخاص بك بطيئًا
     cd scikit-learn

   إذا كنت تخطط لتقديم طلب سحب، يجب أن تستنسخ من شوكتك بدلاً من ذلك.

#. قم بتثبيت إصدار حديث من Python (3.9 أو أحدث في وقت الكتابة) على سبيل المثال باستخدام Miniforge3_. يوفر Miniforge توزيعًا يعتمد على conda لـ Python والمكتبات العلمية الأكثر شيوعًا.

   إذا قمت بتثبيت Python باستخدام conda، نوصي بإنشاء `بيئة conda`_ مخصصة مع جميع تبعيات البناء الخاصة بـ scikit-learn (وتحديدًا NumPy_، SciPy_، Cython_، meson-python_ و Ninja_):

   .. prompt:: bash $

     conda create -n sklearn-env -c conda-forge python numpy scipy cython meson-python ninja

   ليس من الضروري دائمًا ولكن من الأسلم فتح موجه جديد قبل تنشيط بيئة conda المنشأة حديثًا.

   .. prompt:: bash $

     conda activate sklearn-env

#. **بديل لـ conda:** يمكنك استخدام منشآت بديلة لـ Python بشرط أن تكون حديثة بما فيه الكفاية (3.9 أو أعلى في وقت الكتابة). فيما يلي مثال على كيفية إنشاء بيئة بناء لنظام Linux Python. يتم تثبيت تبعيات البناء باستخدام `pip` في virtualenv_ مخصص لتجنب الإخلال ببرامج Python الأخرى المثبتة على النظام:

   .. prompt:: bash $

     python3 -m venv sklearn-env
     source sklearn-env/bin/activate
     pip install wheel numpy scipy cython meson-python ninja

#. قم بتثبيت مترجم مع دعم OpenMP_ لمنصتك. راجع التعليمات الخاصة :ref:`compiler_windows`، :ref:`compiler_macos`، :ref:`compiler_linux` و :ref:`compiler_freebsd`.

#. بناء المشروع مع pip:

   .. prompt:: bash $

     pip install --editable . \
        --verbose --no-build-isolation \
        --config-settings editable-verbose=true

#. تحقق من أن scikit-learn المثبت لديه رقم إصدار ينتهي بـ `.dev0`:

   .. prompt:: bash $

     python -c "import sklearn; sklearn.show_versions()"

#. يرجى الرجوع إلى :ref:`developers_guide` و :ref:`pytest_tips` لتشغيل الاختبارات على الوحدة التي اخترتها.

.. note::

    `--config-settings editable-verbose=true` هو اختياري ولكن يوصى به لتجنب المفاجآت عند استيراد `sklearn`. `meson-python` ينفذ منشآت قابلة للتحرير بإعادة بناء `sklearn` عند تنفيذ `import sklearn`. مع الإعداد الموصى به سترى رسالة عندما يحدث هذا، بدلاً من الانتظار المحتمل دون تغذية راجعة و التساؤل عما يستغرق وقتًا طويلاً. مكافأة: هذا يعني أنه عليك فقط تشغيل أمر `pip install` مرة واحدة، سيتم إعادة بناء `sklearn` تلقائيًا عند استيراد `sklearn`.

التبعيات
------------

تبعيات وقت التشغيل
~~~~~~~~~~~~~~~~~~~~

يتطلب Scikit-learn التبعيات التالية في كل من وقت البناء ووقت التشغيل:

- Python (>= 3.8)،
- NumPy (>= |NumpyMinVersion|)،
- SciPy (>= |ScipyMinVersion|)،
- Joblib (>= |JoblibMinVersion|)،
- threadpoolctl (>= |ThreadpoolctlMinVersion|).

تبعيات البناء
~~~~~~~~~~~~~~~~~~

يتطلب بناء Scikit-learn أيضًا:

..
    # تحتاج الأماكن التالية إلى التوافق فيما يتعلق بإصدار Cython:
    # - ملف تكوين .circleci
    # - sklearn/_build_utils/__init__.py
    # - دليل التثبيت المتقدم

- Cython >= |CythonMinVersion|
- مترجم C/C++ ومكتبة وقت تشغيل OpenMP_ مطابقة. راجع :ref:`platform system specific instructions <platform_specific_instructions>` للحصول على مزيد من التفاصيل.

.. note::

   إذا لم يتم دعم OpenMP من قبل المترجم، فسيتم البناء مع تعطيل وظائف OpenMP. هذا غير موصى به لأنه سيجبر بعض المقدرين على العمل في الوضع التسلسلي بدلاً من الاستفادة من التوازي المستند إلى الخيوط. سيؤدي تعيين متغير البيئة ``SKLEARN_FAIL_NO_OPENMP`` (قبل cythonization) إلى إجبار البناء على الفشل إذا لم يتم دعم OpenMP.

منذ الإصدار 0.21، يكتشف scikit-learn تلقائيًا ويستخدم مكتبة الجبر الخطي التي يستخدمها SciPy **في وقت التشغيل**. لذلك لا يوجد تبعية بناء لـ scikit-learn على تنفيذ BLAS / LAPACK مثل OpenBlas أو Atlas أو Blis أو MKL.

تبعيات الاختبار
~~~~~~~~~~~~~~~~~

يتطلب تشغيل الاختبارات:

- pytest >= |PytestMinVersion|

تتطلب بعض الاختبارات أيضًا `pandas <https://pandas.pydata.org>`_.


بناء إصدار محدد من علامة
--------------------------------------

إذا كنت ترغب في إنشاء إصدار مستقر، يمكنك ``git checkout <VERSION>`` للحصول على الشفرة لهذا الإصدار المحدد، أو تنزيل أرشيف zip للإصدار من github.

.. _platform_specific_instructions:

تعليمات خاصة بالنظام الأساسي


(ملحوظة: في الترجمة العربية، تم الحفاظ على الأسماء الأصلية لبعض الأدوات والتقنيات المذكورة في النص الأصلي، مثل Git و Miniforge3 و conda و pip و OpenMP و NumPy و SciPy و Cython و meson-python و Ninja و pandas و pytest، وذلك بسبب شيوع استخدامها في المجتمع التقني. كما تم الحفاظ على أسماء بعض الأوامر والنصوص البرمجية، مثل git clone و conda create و pip install و python -m venv و python -c، وذلك لضمان الدقة في الترجمة. بالإضافة إلى ذلك، تم الحفاظ على أسماء بعض المتغيرات والعلامات، مثل SKLEARN_FAIL_NO_OPENMP و --config-settings و --verbose و --no-build-isolation و editable-verbose=true، وذلك لضمان الدقة في الترجمة وعدم تغيير معنى النص الأصلي.)

إليك ترجمة النص إلى اللغة العربية، مع مراعاة عدم ترجمة الرموز الخاصة والرياضية والروابط والأكواد البرمجية:

==============================

فيما يلي تعليمات تثبيت مُجمِّع C/C++ يعمل مع دعم OpenMP لبناء إضافات Cython في scikit-learn لكل منصة مدعومة.

.. _compiler_windows:

ويندوز
------

أولاً، حمّل أداة تثبيت "Build Tools for Visual Studio 2019" من الرابط التالي:

<https://aka.ms/vs/17/release/vs_buildtools.exe>

قم بتشغيل ملف "vs_buildtools.exe" الذي حمّلته، وخلال التثبيت، تأكد من تحديد "Desktop development with C++"، كما هو موضح في الصورة التالية:

.. image:: ../images/visual-studio-build-tools-selection.png

ثانياً، تعرف على ما إذا كنت تستخدم إصدار 64-bit أو 32-bit من Python. عملية البناء تعتمد على بنية مُفسر Python. يمكنك التحقق من البنية عن طريق تشغيل الأمر التالي في نافذة "cmd" أو "powershell":

.. prompt:: bash $

    python -c "import struct; print(struct.calcsize('P') * 8)"

بالنسبة لـ 64-bit Python، قم بتكوين بيئة البناء عن طريق تشغيل الأوامر التالية في نافذة "cmd" أو "Anaconda Prompt" (إذا كنت تستخدم Anaconda):

.. sphinx-prompt 1.3.0 (used in doc-min-dependencies CI task) does not support `batch` prompt type,
.. so we work around by using a known prompt type and an explicit prompt text.
..
.. prompt:: bash C:\>

    SET DISTUTILS_USE_SDK=1
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

استبدل "x64" بـ "x86" للبناء لـ 32-bit Python.

يرجى ملاحظة أن المسار السابق قد يختلف من مستخدم لآخر. الهدف هو الإشارة إلى ملف "vcvarsall.bat" الذي سيعيّن متغيرات البيئة الضرورية في موجه الأوامر الحالي.

أخيراً، قم ببناء scikit-learn باستخدام موجه الأوامر هذا:

.. prompt:: bash $

    pip install --editable . \
        --verbose --no-build-isolation \
        --config-settings editable-verbose=true

.. _compiler_macos:

ماك أو إس
-----

المُجمِّع الافتراضي لـ C على ماك أو إس، Apple clang (المعروف باسم `/usr/bin/gcc`)، لا يدعم OpenMP مباشرة. نقدم طريقتين لتمكين دعم OpenMP:

- إما تثبيت `conda-forge::compilers` باستخدام conda؛

- أو تثبيت `libomp` باستخدام Homebrew لتوسيع مُجمِّع Apple clang الافتراضي.

بالنسبة لأجهزة Apple Silicon M1، الطريقة الوحيدة المعروفة أنها تعمل حالياً (يناير 2021) هي طريقة conda-forge. يمكنك تثبيت توزيع `macos/arm64` الخاص بـ conda باستخدام `miniforge installer
<https://github.com/conda-forge/miniforge#miniforge>`_

مُجمِّعات ماك أو إس من conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

إذا كنت تستخدم مدير الحزم conda (الإصدار >= 4.7)، يمكنك تثبيت الحزمة "compilers" من قناة conda-forge، والتي توفر مُجمِّعات C/C++ تدعم OpenMP بناءً على أداة llvm.

قم أولاً بتثبيت أدوات سطر أوامر ماك أو إس:

.. prompt:: bash $

    xcode-select --install

من المستحسن استخدام `conda environment`_ مخصص لبناء scikit-learn من المصدر:

.. prompt:: bash $

    conda create -n sklearn-dev -c conda-forge python numpy scipy cython \
        joblib threadpoolctl pytest compilers llvm-openmp meson-python ninja

ليس من الضروري دائماً، ولكن من الأفضل فتح موجه أوامر جديد قبل تفعيل بيئة conda التي تم إنشاؤها حديثاً.

.. prompt:: bash $

    conda activate sklearn-dev
    make clean
    pip install --editable . \
        --verbose --no-build-isolation \
        --config-settings editable-verbose=true

.. note::

    إذا واجهت أي رسالة خطأ لتعارض في التبعيات، جرب تعليق أي إعدادات مخصصة لـ conda في ملف ``$HOME/.condarc``. على وجه الخصوص، فإن توجيه ``channel_priority: strict`` معروف أنه يسبب مشاكل لهذه العملية.

يمكنك التأكد من تثبيت المُجمِّعات المخصصة بشكل صحيح من conda forge باستخدام الأمر التالي:

.. prompt:: bash $

    conda list

والذي يجب أن يتضمن ``compilers`` و ``llvm-openmp``.

ستقوم حزمة compilers تلقائياً بتعيين متغيرات بيئة مخصصة:

.. prompt:: bash $

    echo $CC
    echo $CXX
    echo $CFLAGS
    echo $CXXFLAGS
    echo $LDFLAGS

وتشير إلى ملفات ومجلدات من بيئة ``sklearn-dev`` الخاصة بك (على وجه الخصوص في المجلدات الفرعية bin/ و include/ و lib/). على سبيل المثال، يجب أن يظهر ``-L/path/to/conda/envs/sklearn-dev/lib`` في ``LDFLAGS``.

في السجل، يجب أن ترى امتداد المُجمَّع الذي تم بناؤه باستخدام مُجمِّعات clang و clang++ المثبتة بواسطة conda مع علامة سطر الأوامر ``-fopenmp``.

مُجمِّعات ماك أو إس من Homebrew
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

حل آخر هو تمكين دعم OpenMP لمُجمِّع clang الذي يتم شحنه افتراضياً على ماك أو إس.

قم أولاً بتثبيت أدوات سطر أوامر ماك أو إس:

.. prompt:: bash $

    xcode-select --install

ثبت مدير الحزم Homebrew_ لماك أو إس.

ثبت مكتبة LLVM OpenMP:

.. prompt:: bash $

    brew install libomp

عيّن متغيرات البيئة التالية:

.. prompt:: bash $

    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
    export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
    export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
    export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib -L/usr/local/opt/libomp/lib -lomp"

أخيراً، قم ببناء scikit-learn في وضع اللفظي (لفحص وجود علامة ``-fopenmp`` في أوامر المُجمِّع):

.. prompt:: bash $

    make clean
    pip install --editable . \
        --verbose --no-build-isolation \
        --config-settings editable-verbose=true

.. _compiler_linux:

لينكس

    
    (نهاية الترجمة)

  هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية، مع الحفاظ على الرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية كما هي:

-----

مترجمات لينكس من النظام
~~~~~~~~~~~~~~~~~~~~~~~~

يتطلب تثبيت scikit-learn من المصدر دون استخدام conda أن يكون لديك تثبيت رؤوس تطوير Python لـ scikit-learn ومترجم C/C++ عامل مع دعم OpenMP (عادةً ما تكون مجموعة أدوات GCC).

ثبت تبعيات البناء لأنظمة التشغيل المستندة إلى Debian، مثل Ubuntu:

.. prompt:: bash $

    sudo apt-get install build-essential python3-dev python3-pip

ثم استمر كالمعتاد:

.. prompt:: bash $

    pip3 install cython
    pip3 install --editable . \
        --verbose --no-build-isolation \
        --config-settings editable-verbose=true

يجب تثبيت Cython وعجلات ما قبل الترجمة لتبعيات وقت التشغيل (numpy و scipy و joblib) تلقائيًا في ``$HOME/.local/lib/pythonX.Y/site-packages``. بدلاً من ذلك، يمكنك تشغيل الأوامر المذكورة أعلاه من virtualenv_ أو `بيئة conda`_ للحصول على عزل كامل عن حزم Python المثبتة عبر النظام. عند استخدام بيئة معزولة، يجب استبدال ``pip3`` بـ ``pip`` في الأوامر المذكورة أعلاه.

عندما لا تتوفر عجلات ما قبل الترجمة لتبعيات وقت التشغيل للهندسة المعمارية الخاصة بك (مثل ARM)، يمكنك تثبيت إصدارات النظام:

.. prompt:: bash $

    sudo apt-get install cython3 python3-numpy python3-scipy

على Red Hat والمشتقات (مثل CentOS)، ثبت التبعيات باستخدام:

.. prompt:: bash $

    sudo yum -y install gcc gcc-c++ python3-devel numpy scipy

مترجمات لينكس من conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

بدلاً من ذلك، ثبت إصدار حديث من مجموعة أدوات GNU C Compiler (GCC) في مجلد المستخدم باستخدام conda:

.. prompt:: bash $

    conda create -n sklearn-dev -c conda-forge python numpy scipy cython \
        joblib threadpoolctl pytest compilers meson-python ninja

ليس من الضروري دائمًا، ولكن من الآمن فتح موجه جديد قبل تنشيط بيئة conda المنشأة حديثًا.

.. prompt:: bash $

    conda activate sklearn-dev
    pip install --editable . \
        --verbose --no-build-isolation \
        --config-settings editable-verbose=true

.. _compiler_freebsd:

FreeBSD
-------

لا يتضمن مترجم clang المضمن في أنظمة FreeBSD 12.0 و 11.2 الأساسية دعم OpenMP. تحتاج إلى تثبيت مكتبة `openmp` من الحزم (أو المنافذ):

.. prompt:: bash $

    sudo pkg install openmp

سيؤدي هذا إلى تثبيت ملفات الرأس في ``/usr/local/include`` والمكتبات في ``/usr/local/lib``. نظرًا لأن هذه الدلائل لا يتم البحث عنها افتراضيًا، يمكنك تعيين متغيرات البيئة إلى هذه المواقع:

.. prompt:: bash $

    export CFLAGS="$CFLAGS -I/usr/local/include"
    export CXXFLAGS="$CXXFLAGS -I/usr/local/include"
    export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/lib -L/usr/local/lib -lomp"

أخيرًا، قم ببناء الحزمة باستخدام الأمر القياسي:

.. prompt:: bash $

    pip install --editable . \
        --verbose --no-build-isolation \
        --config-settings editable-verbose=true

بالنسبة لإصدارات FreeBSD 12.1 و 11.3 القادمة، سيتم تضمين OpenMP في النظام الأساسي ولن تكون هذه الخطوات ضرورية.

.. _OpenMP: https://en.wikipedia.org/wiki/OpenMP
.. _Cython: https://cython.org
.. _meson-python: https://mesonbuild.com/meson-python
.. _Ninja: https://ninja-build.org/
.. _NumPy: https://numpy.org
.. _SciPy: https://www.scipy.org
.. _Homebrew: https://brew.sh
.. _virtualenv: https://docs.python.org/3/tutorial/venv.html
.. _conda environment: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
.. _Miniforge3: https://github.com/conda-forge/miniforge#miniforge3
