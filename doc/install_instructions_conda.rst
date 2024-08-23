  هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية، مع الحفاظ على الرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية دون ترجمة:

قم بتثبيت conda باستخدام "Anaconda أو miniforge installers" من خلال الرابط التالي:

<https://docs.conda.io/projects/conda/en/latest/user-guide/install/>

أو باستخدام "miniforge installers" من خلال الرابط التالي (لا يتطلب أيًا منهما صلاحيات مدير النظام):

<https://github.com/conda-forge/miniforge#miniforge>

ثم نفّذ الأوامر التالية:

.. prompt:: bash

 conda create -n sklearn-env -c conda-forge scikit-learn
 conda activate sklearn-env

للتحقق من التثبيت، يمكنك استخدام الأوامر التالية:

.. prompt:: bash

 conda list scikit-learn  # عرض إصدار scikit-learn وموقعه
 conda list               # عرض جميع الحزم المثبتة في البيئة
 python -c "import sklearn; sklearn.show_versions()"
