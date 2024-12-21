
import 'package:flutter/material.dart';
import 'package:zen_tsyp_app/widgets/clothing_item.dart';


class ManCothingScreen extends StatefulWidget {
  static const routeName = '/Choose_clothing_screen';
  const ManCothingScreen({super.key});

  @override
  State<ManCothingScreen> createState() => _ManCothingScreenState();
}

class _ManCothingScreenState extends State<ManCothingScreen> {
  final List<ClothingItem> clothesList = [
   const  ClothingItem(title: 'Costume',imageAsset: 'assets/costume.jfif',price: 120,size: 'XL',description: 'Crafted with precision, this costume combines soft, durable fabric with a tailored fit, perfect for casual or formal wear',),
   const  ClothingItem(title: 'Man T-shirt',imageAsset: 'assets/men-shirt.jpg',price: 50,size: 'L',description: 'Crafted with precision, this shirt combines soft, durable fabric with a tailored fit, perfect for casual or formal wear',),
    const ClothingItem(title:'Pants' ,imageAsset: 'assets/man-1.jfif', price: 80, size: 'S',description: 'These versatile pants offer a perfect blend of comfort and style, featuring a tailored fit and premium fabric for any occasion.',),
    const ClothingItem(title:'Shirt' ,imageAsset: 'assets/man-2.jfif', price: 60, size: 'M' ,description: 'Experience ultimate comfort and sophistication with this classic shirt, designed to elevate your wardrobe effortlessly',),
    const ClothingItem(title:'Shirt' ,imageAsset: 'assets/man-4.jfif', price: 130, size: 'S',description: 'Made from high-quality materials, this shirt boasts a modern design with attention to detail, making it a versatile wardrobe staple',),
    const ClothingItem(title:'Pants' ,imageAsset: 'assets/man-6.jfif', price: 75, size: 'M', description: 'Crafted for durability and elegance, these pants provide a sleek design with a comfortable feel, ideal for work or leisure.',),
    const ClothingItem(title: 'Shirt',imageAsset: 'assets/man-7.jpg',price: 40,size: 'L',description: 'This coat combines timeless elegance with superior warmth, crafted from premium materials to keep you stylish and comfortable in colder weather',),
    const ClothingItem(title: 'Costume',imageAsset: 'assets/man-8.jpg',price: 57,size: 'XL',description: 'Designed for both function and fashion, this coat features a tailored fit, durable fabric, and thoughtful details for a sophisticated look',),
    const ClothingItem(title: 'Shirt',imageAsset: 'assets/man-9.png',price: 90,size: 'L',description: 'A classic coat that blends luxurious texture with modern design, offering versatile wearability for any occasion',),


  ];
  final itemNameController = TextEditingController();
  final itemSize =TextEditingController();
  final itemPrice = TextEditingController();
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        padding: const EdgeInsets.all(10),
        child:  GridView.builder(
          itemCount: clothesList.length,
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 2,
          childAspectRatio: 3/2,
          crossAxisSpacing: 10,
          mainAxisSpacing: 10
        ), itemBuilder: (ctx,i){
          return clothesList[i];
        }),
      )
    );
  }
}