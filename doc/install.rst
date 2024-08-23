=======
تثبيت scikit-learn
=======

هناك عدة طرق مختلفة لتثبيت حزمة scikit-learn:

* :ref:`تثبيت الإصدار الرسمي الأخير <install_official_release>`. هذه هي أفضل طريقة لمعظم المستخدمين، حيث ستوفر إصدارًا مستقرًا وتتوفر حزم جاهزة لمعظم المنصات.

* تثبيت إصدار scikit-learn المقدم من :ref:`نظام التشغيل الخاص بك أو توزيع بايثون الذي تستخدمه <install_by_distribution>`. هذه طريقة سريعة لأولئك الذين لديهم أنظمة تشغيل أو توزيعات بايثون توزع scikit-learn. ولكن قد لا توفر أحدث إصدار.

* :ref:`بناء الحزمة من المصدر <install_bleeding_edge>`. هذه الطريقة الأفضل للمستخدمين الذين يرغبون في الحصول على أحدث الميزات والذين لا يخشون تشغيل التعليمات البرمجية الجديدة. كما أنها مطلوبة للمستخدمين الذين يرغبون في المساهمة في المشروع.

.. _install_official_release:

تثبيت الإصدار الرسمي الأخير
    
(ملاحظة: يرجى العلم أنني لم أقم بترجمة الرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية كما طلبت في السؤال.)

هذا نص بتنسيق RST الذي تريد ترجمته إلى اللغة العربية. سأحرص على عدم ترجمة الرموز الخاصة، والرموز والمعادلات الرياضية، والروابط والتاجات، والشفرة البرمجية.

=============================

.. raw:: html

  <style>
    /* Show caption on large screens */
    @media screen and (min-width: 960px) {
      .install-instructions .sd-tab-set {
        --tab-caption-width: 20%;
      }

      .install-instructions .sd-tab-set.tabs-os::before {
        content: "نظام التشغيل";
      }

      .install-instructions .sd-tab-set.tabs-package-manager::before {
        content: "مدير الحزم";
      }
    }
  </style>

.. div:: install-instructions

  .. tab-set::
    :class: tabs-os

    .. tab-item:: Windows
      :class-label: tab-4

      .. tab-set::
        :class: tabs-package-manager

        .. tab-item:: pip
          :class-label: tab-6
          :sync: package-manager-pip

          قم بتثبيت الإصدار 64-بت من Python 3، على سبيل المثال من
          `الموقع الرسمي <https://www.python.org/downloads/windows/>`__.

          الآن، قم بإنشاء `بيئة افتراضية (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ وتثبيت scikit-learn.
          لاحظ أن البيئة الافتراضية خيارية ولكن ينصح بها بشدة، لتجنب التعارضات المحتملة مع الحزم الأخرى.

          .. prompt:: powershell

            python -m venv sklearn-env
            sklearn-env\Scripts\activate  # تنشيط
            pip install -U scikit-learn

          للتحقق من التثبيت، يمكنك استخدام:

          .. prompt:: powershell

            python -m pip show scikit-learn  # إظهار إصدار scikit-learn والموقع
            python -m pip freeze             # إظهار جميع الحزم المثبتة في البيئة
            python -c "import sklearn; sklearn.show_versions()"

        .. tab-item:: conda
          :class-label: tab-6
          :sync: package-manager-conda

          .. include:: ./install_instructions_conda.rst

    .. tab-item:: MacOS
      :class-label: tab-4

      .. tab-set::
        :class: tabs-package-manager

        .. tab-item:: pip
          :class-label: tab-6
          :sync: package-manager-pip

          قم بتثبيت Python 3 باستخدام `homebrew <https://brew.sh/>`_ (`brew install python`)
          أو عن طريق تثبيت الحزمة يدويًا من `الموقع الرسمي
          <https://www.python.org/downloads/macos/>`__.

          الآن، قم بإنشاء `بيئة افتراضية (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ وتثبيت scikit-learn.
          لاحظ أن البيئة الافتراضية خيارية ولكن ينصح بها بشدة، لتجنب التعارضات المحتملة مع الحزم الأخرى.

          .. prompt:: bash

            python -m venv sklearn-env
            source sklearn-env/bin/activate  # تنشيط
            pip install -U scikit-learn

          للتحقق من التثبيت، يمكنك استخدام:

          .. prompt:: bash

            python -m pip show scikit-learn  # إظهار إصدار scikit-learn والموقع
            python -m pip freeze             # إظهار جميع الحزم المثبتة في البيئة
            python -c "import sklearn; sklearn.show_versions()"

        .. tab-item:: conda
          :class-label: tab-6
          :sync: package-manager-conda

          .. include:: ./install_instructions_conda.rst

    .. tab-item:: Linux
      :class-label: tab-4

      .. tab-set::
        :class: tabs-package-manager

        .. tab-item:: pip
          :class-label: tab-6
          :sync: package-manager-pip

          عادةً ما يكون Python 3 مثبتًا بشكل افتراضي على معظم توزيعات Linux. للتحقق مما إذا كان لديك Python 3 مثبتًا، جرّب:

          .. prompt:: bash

            python3 --version
            pip3 --version

          إذا لم يكن لديك Python 3 مثبتًا، يرجى تثبيت `python3` و
          `python3-pip` من مدير الحزم الخاص بالتوزيعة.

          الآن، قم بإنشاء `بيئة افتراضية (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ وتثبيت scikit-learn.
          لاحظ أن البيئة الافتراضية خيارية ولكن ينصح بها بشدة، لتجنب التعارضات المحتملة مع الحزم الأخرى.

          .. prompt:: bash

            python3 -m venv sklearn-env
            source sklearn-env/bin/activate  # تنشيط
            pip3 install -U scikit-learn

          للتحقق من التثبيت، يمكنك استخدام:

          .. prompt:: bash

            python3 -m pip show scikit-learn  # إظهار إصدار scikit-learn والموقع
            python3 -m pip freeze             # إظهار جميع الحزم المثبتة في البيئة
            python3 -c "import sklearn; sklearn.show_versions()"

        .. tab-item:: conda
          :class-label: tab-6
          :sync: package-manager-conda

          .. include:: ./install_instructions_conda.rst


استخدام بيئة معزولة مثل pip venv أو conda يجعل من الممكن
تثبيت إصدار محدد من scikit-learn مع pip أو conda وتبعياتها
بشكل مستقل عن أي حزم Python مثبتة مسبقًا. بشكل خاص في Linux
ينصح بعدم تثبيت حزم pip بجانب الحزم المدارة بواسطة
مدير الحزم الخاص بالتوزيعة (apt، dnf، pacman...).

لاحظ أنه يجب عليك دائمًا تذكر تنشيط البيئة التي اخترتها
قبل تشغيل أي أمر Python كلما تبدأ جلسة طرفية جديدة.

إذا لم تقم بتثبيت NumPy أو SciPy بعد، يمكنك أيضًا تثبيتها باستخدام
conda أو pip. عند استخدام pip، يرجى التأكد من استخدام *binary wheels*،
وأن NumPy و SciPy لا يتم إعادة ترجمتها من المصدر، والتي يمكن أن تحدث عند استخدام
تكوينات معينة من نظام التشغيل والأجهزة (مثل Linux على
Raspberry Pi).

تتطلب إمكانيات التخطيط في Scikit-learn (أي، الدوال التي تبدأ بـ `plot\_`
والفئات التي تنتهي بـ `Display`) Matplotlib. تتطلب الأمثلة
Matplotlib وبعض الأمثلة تتطلب scikit-image، أو pandas، أو seaborn.
يتم سرد الحد الأدنى لإصدار تبعيات scikit-learn أدناه مع الغرض منها.

.. include:: min_dependency_table.rst

.. warning::

    

(ملاحظة: لم تتم ترجمة محتوى الملف المضمن `min_dependency_table.rst` حيث أنه غير متوفر في النص الأصلي.)

هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية، مع الحفاظ على الرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية:

كانت Scikit-learn 0.20 آخر إصدار يدعم بايثون 2.7 وبايثون 3.4.

دعم Scikit-learn 0.21 بايثون 3.5-3.7.

دعم Scikit-learn 0.22 بايثون 3.5-3.8.

تطلب Scikit-learn 0.23-0.24 بايثون 3.6 أو أحدث.

دعم Scikit-learn 1.0 بايثون 3.7-3.10.

يدعم Scikit-learn 1.1 و1.2 و1.3 بايثون 3.8-3.12

يحتاج Scikit-learn 1.4 إلى بايثون 3.9 أو أحدث.

.. _install_by_distribution:

التوزيعات الخارجية لـ Scikit-learn

(المقصود بالتوزيعات الخارجية هي التوزيعات التي توفرها جهات خارجية وليست التوزيعات الرسمية)

هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية. لا تترجم الرموز الخاصة ولا الرموز والمعادلات الرياضية ولا تترجم الروابط والتاجات ولا الشفرة البرمجية:

=========================================

توفر بعض توزيعات الجهات الخارجية إصدارات من scikit-learn مدمجة مع أنظمة إدارة الحزم الخاصة بها.

يمكن أن تجعل هذه التوزيعات عملية التثبيت والترقية أسهل بكثير للمستخدمين، نظرًا لأن التكامل يشمل القدرة على تثبيت التبعيات (numpy، scipy) التي يتطلبها scikit-learn تلقائيًا.

فيما يلي قائمة غير كاملة بأنظمة التشغيل وتوزيعات Python التي توفر إصدارها الخاص من scikit-learn.

Alpine Linux
------------

يتم توفير حزمة Alpine Linux من خلال `المستودعات الرسمية <https://pkgs.alpinelinux.org/packages?name=py3-scikit-learn>`__ باسم ``py3-scikit-learn`` لـ Python.
يمكن تثبيتها عن طريق كتابة الأمر التالي:

.. prompt:: bash

 sudo apk add py3-scikit-learn


Arch Linux
----------

يتم توفير حزمة Arch Linux من خلال `المستودعات الرسمية <https://www.archlinux.org/packages/?q=scikit-learn>`_ باسم ``python-scikit-learn`` لـ Python.
يمكن تثبيتها عن طريق كتابة الأمر التالي:

.. prompt:: bash

 sudo pacman -S python-scikit-learn


Debian/Ubuntu
-------------

تُقسم حزمة Debian/Ubuntu إلى ثلاث حزم مختلفة تسمى ``python3-sklearn`` (وحدات python)، و ``python3-sklearn-lib`` (عمليات التنفيذ منخفضة المستوى والارتباطات)، و ``python3-sklearn-doc`` (الوثائق).
لاحظ أن scikit-learn يتطلب Python 3، وبالتالي الحاجة إلى استخدام أسماء الحزم ذات البادئة `python3-`.
يمكن تثبيت الحزم باستخدام ``apt-get``:

.. prompt:: bash

 sudo apt-get install python3-sklearn python3-sklearn-lib python3-sklearn-doc


Fedora
------

تسمى حزمة Fedora ``python3-scikit-learn`` لإصدار Python 3، وهو الإصدار الوحيد المتوفر في Fedora.
يمكن تثبيتها باستخدام ``dnf``:

.. prompt:: bash

 sudo dnf install python3-scikit-learn


NetBSD
------

يتوفر scikit-learn عبر `pkgsrc-wip <http://pkgsrc-wip.sourceforge.net/>`_:
https://pkgsrc.se/math/py-scikit-learn


MacPorts for Mac OSX
--------------------

تسمى حزمة MacPorts ``py<XY>-scikits-learn``،
حيث ``XY`` تدل على إصدار Python.
يمكن تثبيتها عن طريق كتابة الأمر التالي:

.. prompt:: bash

 sudo port install py39-scikit-learn


Anaconda و Enthought Deployment Manager لجميع الأنظمة الأساسية المدعومة
---------------------------------------------------------------------

كل من `Anaconda <https://www.anaconda.com/download>`_ و
`Enthought Deployment Manager <https://assets.enthought.com/downloads/>`_
يشحنان scikit-learn بالإضافة إلى مجموعة كبيرة من مكتبات Python العلمية لنظامي التشغيل Windows و Mac OSX و Linux.

تقدم Anaconda scikit-learn كجزء من توزيعها المجاني.


Intel Extension for Scikit-learn
--------------------------------

تحافظ Intel على حزمة x86_64 محسّنة، متوفرة في PyPI (عبر `pip`)،
وفي قنوات `main` و `conda-forge` و `intel` conda:

.. prompt:: bash

 conda install scikit-learn-intelex

تحتوي هذه الحزمة على إصدار محسّن من Intel للعديد من المُقدّرين. عندما
لا يوجد إصدار بديل، يتم استخدام تنفيذ scikit-learn كخيار احتياطي. تأتي
هذه الحلول المحسّنة من مكتبة oneDAL بلغة C ++ وهي مُحسّنة
لبنية x86_64، وهي مُحسّنة لوحدات المعالجة المركزية Intel متعددة النواة.

لاحظ أن هذه الحلول ليست ممكّنة افتراضيًا، يرجى الرجوع إلى
`scikit-learn-intelex <https://intel.github.io/scikit-learn-intelex/latest/what-is-patching.html>`_
الوثائق لمزيد من التفاصيل حول سيناريوهات الاستخدام. مثال التصدير المباشر:

.. prompt:: python >>>

 from sklearnex.neighbors import NearestNeighbors

يتم التحقق من التوافق مع حلول scikit-learn القياسية عن طريق تشغيل مجموعة اختبار
scikit-learn الكاملة عبر التكامل المستمر الآلي كما هو مذكور في
https://github.com/intel/scikit-learn-intelex. إذا كنت تلاحظ أي مشكلة
مع `scikit-learn-intelex`، يرجى الإبلاغ عن المشكلة في
`متعقب المشكلات <https://github.com/intel/scikit-learn-intelex/issues>`__.


WinPython for Windows
---------------------

يقوم مشروع `WinPython <https://winpython.github.io/>`_ بتوزيع
scikit-learn كإضافة إضافية.


استكشاف الأخطاء وإصلاحها
=======================

إذا واجهت حالات فشل غير متوقعة عند تثبيت scikit-learn، فيمكنك إرسال مشكلة إلى
`متعقب المشكلات <https://github.com/scikit-learn/scikit-learn/issues>`_.
قبل ذلك، يرجى أيضًا التأكد من التحقق من المشكلات الشائعة التالية.

.. _windows_longpath:

خطأ ناتج عن حد طول مسار الملف في Windows
---------------------------------------

قد يحدث أن تفشل pip في تثبيت الحزم عند الوصول إلى حد حجم المسار الافتراضي لـ Windows إذا تم تثبيت Python في موقع متداخل مثل
بنية مجلد `AppData` تحت دليل المستخدم الرئيسي، على سبيل المثال::

    C:\Users\username>C:\Users\username\AppData\Local\Microsoft\WindowsApps\python.exe -m pip install scikit-learn
    Collecting scikit-learn
    ...
    Installing collected packages: scikit-learn
    ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: 'C:\\Users\\username\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python37\\site-packages\\sklearn\\datasets\\tests\\data\\openml\\292\\api-v1-json-data-list-data_name-australian-limit-2-data_version-1-status-deactivated.json.gz'

في هذه الحالة، من الممكن رفع هذا الحد في سجل Windows باستخدام
أداة ``regedit``:

#. اكتب "regedit" في قائمة ابدأ في Windows لإطلاق ``regedit``.

#. انتقل إلى مفتاح
   ``Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem``.

#. قم بتحرير قيمة خاصية ``LongPathsEnabled`` لهذا المفتاح وقم بتعيينها على 1.

#. أعد تثبيت scikit-learn (مع تجاهل التثبيت المكسور السابق):

   .. prompt:: powershell

      pip install --exists-action=i scikit-learn
    
  

(ملحوظة: لم يتم ترجمة أسماء الحزم والروابط والتاجات والشفرة البرمجية طبقاً لشروط الترجمة)
