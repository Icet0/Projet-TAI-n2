import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart';
// import 'package:google_ml_kit/google_ml_kit.dart';

class CameraPage extends StatefulWidget {
  const CameraPage({Key? key}) : super(key: key);

  @override
  _CameraPageState createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  late CameraController _cameraController;
  late List<CameraDescription> _cameras;
  late Future<void> _initializeControllerFuture;

  bool _isLoading = false;
  Uint8List? _image;
  String? _myName;
  Uint8List? _imageBytes = Uint8List(8);
  List<int>? imageBytes;

  // final FaceDetector _faceDetector = GoogleMlKit.vision.faceDetector();
  // List<Face>? _faces;

  @override
  void initState() {
    super.initState();

    // Récupérer la liste des caméras disponibles sur l'appareil
    availableCameras().then((cameras) {
      setState(() {
        _cameras = cameras;
      });
      // Initialiser le contrôleur de caméra avec la première caméra disponible
      if (kIsWeb) {
        // app is running on a web browser
        _cameraController =
            CameraController(_cameras[0], ResolutionPreset.high);
      } else {
        // app is running on a mobile device
        _cameraController =
            CameraController(_cameras[1], ResolutionPreset.high);
      }
      // Préparer le contrôleur de caméra pour l'utilisation
      _initializeControllerFuture = _cameraController.initialize();
    });
  }

  @override
  void dispose() {
    // Assurez-vous de libérer les ressources de la caméra lorsque la page est fermée
    _cameraController.dispose();
    // _faceDetector.close();

    super.dispose();
  }

  // void _detectFaces(InputImage image) async {
  //   final faces = await _faceDetector.processImage(image);
  //   setState(() {
  //     _faces = faces;
  //   });
  // }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _image == null && _isLoading == false
          ? FutureBuilder<void>(
              future: _initializeControllerFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.done) {
                  // Si le contrôleur de caméra a été initialisé avec succès, afficher la vue de la caméra
                  return Stack(
                    children: [
                      CameraPreview(_cameraController),
                      // _cameraController.value.isInitialized && _faces != null
                      //     ? CustomPaint(
                      //         painter:
                      //             FaceDetectorPainter(_faces!, _detectFaces),
                      //       )
                      //     : Container(),
                    ],
                  );
                } else {
                  // Sinon, afficher un message d'attente
                  return const Center(child: CircularProgressIndicator());
                }
              },
            )
          : Center(
              child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                _isLoading
                    ? CircularProgressIndicator()
                    : _image == null
                        ? Text('Aucune image sélectionnée.')
                        : Expanded(
                            child: kIsWeb
                                ? Image.memory(_imageBytes!)
                                : Image.memory(_image!),
                          ),
                _myName != null
                    ? Text("Vous ressemblez à $_myName")
                    : Container(),
              ],
            )),
      floatingActionButton: FloatingActionButton(
        child: const Icon(Icons.camera_alt),
        // Lorsque le bouton est cliqué, prendre une photo avec la caméra
        onPressed: () async {
          try {
            // Assurez-vous que le contrôleur de caméra a été correctement initialisé
            await _initializeControllerFuture;
            // Prendre une photo avec la caméra et récupérer le chemin du fichier de l'image
            XFile path2 = await _cameraController.takePicture();
            _image = File(path2.path).readAsBytesSync();
            // final imageInput = InputImage.fromFile(_image!);
            setState(() {
              _isLoading = true;
              // _detectFaces(imageInput);
            });

            //REQUEST

            var response;
            try {
              var headers = {
                "Content-type": "application/json",
                "Referer": kIsWeb
                    ? "http://127.0.0.1:5000"
                    : "https://376c-93-29-103-44.eu.ngrok.io",
              };
              var url;
              if (kIsWeb) {
                url = Uri.http("127.0.0.1:5000", "/prediction");
              } else {
                url = Uri.https("376c-93-29-103-44.eu.ngrok.io", "/prediction");
              }
              print("url: $url");
              // response = await http.post(url, headers: headers, body: json);
              // _imageBytes = await _image?.readAsBytes();
              var f = await path2.readAsBytes();
              _imageBytes = f;
              imageBytes = _imageBytes as List<int>;
              String base64Image = base64Encode(imageBytes!);
              var json = jsonEncode({"path": base64Image});
              print("json: $json");
              var request = Request("POST", url)
                ..headers.addAll(headers)
                ..body = json;
              var client = Client();
              response = await client.send(request);
              print("response: $response");
              if (response.statusCode == 200) {
                var responseBody = await response.stream.bytesToString();
                var data = jsonDecode(responseBody);
                var status = data["status"];
                var similarity = data["data"]["similarity"];
                var name = data["data"]["nom"];
                var full_path = data["data"]["image"];
                _imageBytes = base64Decode(full_path);
                _image = _imageBytes;
                setState(() {
                  // _image == null;
                  _myName = name;
                });
              } else {
                var data = jsonDecode(response.body);
                var status = data["status"];
                var error = data["data"];
                print("status: $status, error: $error");
              }
            } catch (e) {
              print(e);
            } finally {
              setState(() {
                _isLoading = false;
              });
            }
          } catch (e) {
            // Gérer les erreurs potentielles ici
            print(e);
          }
        },
      ),
    );
  }
}

// class FaceDetectorPainter extends CustomPainter {
//   FaceDetectorPainter(this.faces, this.detectFaces);
//   final List<Face> faces;
//   final Function detectFaces;

//   @override
//   void paint(Canvas canvas, Size size) {
//     detectFaces();
//     for (Face face in faces) {
//       final rect = face.boundingBox;
//       final paint = Paint()
//         ..style = PaintingStyle.stroke
//         ..strokeWidth = 8.0
//         ..color = Colors.red;
//       canvas.drawRect(rect, paint);
//     }
//   }

//   @override
//   bool shouldRepaint(FaceDetectorPainter oldDelegate) {
//     return oldDelegate.faces != faces;
//   }
// }
