import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart';
import 'package:image_picker/image_picker.dart';

class ImagePickerView extends StatefulWidget {
  @override
  _ImagePickerViewState createState() => _ImagePickerViewState();
}

class _ImagePickerViewState extends State<ImagePickerView> {
  Uint8List? _image;
  Uint8List? _imageBytes = Uint8List(8);

  bool _isLoading = false;
  Uint8List? _imageUrl;
  late List<int> imageBytes;
  String? _myName;

  // _ImagePickerViewState() {
  //   _image = File('');
  // }
  Future getImage() async {
    if (!kIsWeb) {
      final ImagePicker _picker = ImagePicker();
      XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      if (image != null) {
        var selected = File(image.path);
        setState(() {
          _image = File(image.path).readAsBytesSync();
          _imageUrl = _image;
        });
      } else {
        print('No image selected.');
      }
    } else if (kIsWeb) {
      final ImagePicker _picker = ImagePicker();
      XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      if (image != null) {
        var f = await image.readAsBytes();
        setState(() {
          _imageBytes = f;
          _image = File('a').readAsBytesSync();
          _imageUrl = _image;
        });
      } else {
        print('No image selected.');
      }
    } else {
      print("something went wrong");
    }
  }

  void _uploadImage() async {
    setState(() {
      _isLoading = true;
    });
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

      if (!kIsWeb) {
        var imageFile = _imageUrl;
        imageBytes = imageFile as List<int>;
      } else {
        imageBytes = _imageBytes as List<int>;
      }
      String base64Image = base64Encode(imageBytes);
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
        // print(
        //     "status: $status, similarity: $similarity, name: $name, full_path: $full_path");
        _imageBytes = base64Decode(full_path);
        _image = _imageBytes;
        setState(() {
          _imageUrl = null;
          _myName = name.toString();
        });

        // _image = Image.memory(_imageBytes!) as File?;
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
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
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
          _imageUrl != null
              ? Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: ElevatedButton(
                    onPressed: _uploadImage,
                    child: Text("Upload Image"),
                  ),
                )
              : _myName != null
                  ? Text("Vous ressemblez à $_myName")
                  : Container(),
        ],
      )),
      floatingActionButton: FloatingActionButton(
        onPressed: getImage,
        tooltip: 'Sélectionner une image',
        child: Icon(Icons.add_a_photo),
      ),
    );
  }
}
