# Qualcomm QAIRT licensing reference

This file is a reference, not a replacement for Qualcomm's agreement.

The QAIRT/QNN runtime objects under
`app/src/main/jniLibs/arm64-v8a/` are proprietary Qualcomm Technologies,
Inc. software. The controlling agreement is the **AI Stack License** supplied
as `LICENSE.pdf` with the Qualcomm AI Runtime SDK package from which the
runtime objects were obtained.

That agreement permits the QAIRT software to be distributed and sublicensed
in object-code form when it is incorporated into an application. It does not
permit distributing the QAIRT software as a standalone SDK or standalone
runtime package. It also prohibits removing or altering Qualcomm copyright,
proprietary-information, and restricted-rights notices.

Accordingly, the QAIRT objects in this project are provided only as runtime
components of the ZipDepth NPU Demo Android application. They remain under
Qualcomm's terms and are excluded from the repository's 0BSD license. Do not
extract or redistribute them as a separate product.

Official references:

- [Qualcomm AI Engine Direct SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)
- [Qualcomm GenieX QAIRT plugin](https://github.com/qualcomm/geniex-qairt-plugin)
- [GenieX QAIRT third-party notices](https://github.com/qualcomm/geniex-qairt-plugin/blob/main/THIRD_PARTY_NOTICES.md)
- [QAIRT Android prebuilt runtime directory](https://github.com/qualcomm/geniex-qairt-plugin/tree/main/third-party/android)

Qualcomm's official GenieX repository is a useful public packaging reference:
it includes Android QAIRT runtime libraries and identifies those prebuilts as
being governed by the QAIRT SDK EULA rather than the repository's BSD license.
