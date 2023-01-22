import 'dart:io';

import 'package:flutter/material.dart';

class DisplayPictureScreen extends StatelessWidget {
  final String imagePath;

  const DisplayPictureScreen({Key? key, required this.imagePath})
      : super(key: key);

  Widget getWidget(imagePath) {
    Widget monWidget;
    try {
      monWidget = Image.file(File(imagePath));
    } catch (e) {
      monWidget = Image.network(imagePath);      
    }
    return monWidget;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: const Text("Affichage de l'image"),
        ),
        // Afficher l'image dans une vue de type Image
        // Utiliser un constructeur Image.network si l'image provient d'une URL
        // Utiliser un constructeur Image.file si l'image provient d'un fichier
        body: getWidget(imagePath));
  }
}
